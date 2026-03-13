import logging
import os
import random
from typing import Annotated, Callable, Literal

import torch
import torch.distributed as dist
import typer
from vllm import SamplingParams

import wandb
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.sft import (
    evaluate_vllm_loop,
    load_policy_into_vllm_instance,
    setup_hf_and_vllm,
)
from cs336_alignment.utils import (
    get_response_log_probs,
    load_MATH,
    masked_normalize,
    tokenize_prompt_and_output,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = "cuda"  # if torch.cuda.is_available() else "cpu"


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rollout_batch_size = len(rollout_responses)
    # n_prompts_per_rollout_batch is the count of prompts for this function call
    n_prompts = rollout_batch_size // group_size

    rewards_list = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        rewards_list.append(reward_fn(response, gt))

    raw_rewards = torch.Tensor([r["reward"] for r in rewards_list])
    rewards = raw_rewards.view(n_prompts, group_size)
    rewards_mean = torch.mean(rewards, dim=-1, keepdim=True)

    if normalize_by_std:
        rewards_std = torch.std(rewards, dim=-1, keepdim=True)
        rewards = (rewards - rewards_mean) / (rewards_std + advantage_eps)
    else:
        rewards = rewards - rewards_mean

    # Collect reward breakdown stats in metadata to avoid redundant reward_fn calls
    format_rewards = [r.get("format_reward", 0.0) for r in rewards_list]
    answer_rewards = [r.get("answer_reward", 0.0) for r in rewards_list]
    n = max(1, len(rewards_list))
    metadata = {
        "avg_total_reward": raw_rewards.mean().item(),
        "avg_format_reward": sum(format_rewards) / n,
        "avg_answer_reward": sum(answer_rewards) / n,
    }

    return (rewards.flatten(), raw_rewards, metadata)


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    # auto boardcasting or using einops
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    # log probs -- exp --> pi
    old_new_ratio = torch.exp(policy_log_probs - old_log_probs)

    part1 = old_new_ratio * advantages
    part2 = torch.clamp(old_new_ratio, 1 - cliprange, 1 + cliprange) * advantages

    # Per-token boolean map of whether it was clipped
    is_clipped = (
        (old_new_ratio > 1 + cliprange) | (old_new_ratio < 1 - cliprange)
    ).float()

    return (-torch.min(part1, part2), {"is_clipped": is_clipped})


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapper that delegates to the appropriate policy gradient loss function above.
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs (batch_size, sequence_length):
            per-token log-probabilities from the policy being trained.
        loss_type:
            One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards, shape (batch_size, 1):
            Required if loss_type == "no_baseline".
        advantages, shape (batch_size, 1):
            Required for "reinforce_with_baseline" and "grpo_clip".
        old_log_probs, shape (batch_size, sequence_length):
            Required for "grpo_clip".
        cliprange:
            Required for "grpo_clip"; scalar ϵ used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss (batch_size, sequence_length):
                per-token loss.
            metadata: dict
                statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        pg_loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return (pg_loss, {})
    elif loss_type == "reinforce_with_baseline":
        pg_loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return (pg_loss, {})
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    elif loss_type == "grpo_no_clip":
        # Off-policy ratio loss WITHOUT clipping (ablation for grpo_off_policy_clip_ablation)
        # Loss = -(pi_theta / pi_theta_old) * advantage, per token
        ratio = torch.exp(policy_log_probs - old_log_probs)
        pg_loss = -(ratio * advantages)
        return (pg_loss, {})
    else:
        raise NotImplementedError


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    # donot use torch.mean, bcuz considering only the elements with mask value 1
    tensor_mask = torch.sum(tensor * mask, dim=dim)
    count = torch.sum(mask, dim=dim)
    return tensor_mask / count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    if loss_type not in ("no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"):
        raise NotImplementedError

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type,
        raw_rewards, advantages, old_log_probs, cliprange,  # ty:ignore[invalid-argument-type]
    )
    if normalize_constant is not None:
        batch_loss = masked_normalize(per_token_loss, response_mask, normalize_constant, dim=-1)
    else:
        batch_loss = masked_mean(per_token_loss, response_mask, dim=-1)

    # mean loss across sequence
    actual_loss = batch_loss.mean()
    loss = actual_loss / gradient_accumulation_steps
    # autograd
    loss.backward()

    # compute clip fraction here if it was provided
    if "is_clipped" in metadata:
        clip_frac = masked_mean(metadata["is_clipped"], response_mask, dim=-1).mean()
        metadata["clip_frac"] = clip_frac

    metadata |= {"policy_loss": actual_loss.detach()}
    return loss, metadata


def generate_and_flatten_rollouts(
    policy,
    vllm_engine,
    questions_batch: list[dict[str, str]],
    sampling_params: SamplingParams,
) -> tuple[list[str], list[str], list[str]]:
    """Generate rollouts via vLLM and flatten the group_size outputs per prompt.

    Returns:
        (repeated_prompts, rollout_responses, repeated_ground_truths)
        each of length n_prompts * group_size.
    """
    load_policy_into_vllm_instance(policy, vllm_engine)

    prompts = [q["prompt"] for q in questions_batch]
    ground_truths = [str(q.get("ground_truth", "")) for q in questions_batch]

    outputs = vllm_engine.generate(prompts, sampling_params)

    rollout_responses = []
    repeated_ground_truths = []
    repeated_prompts = []
    for output, gt, pr in zip(outputs, ground_truths, prompts):
        for out in output.outputs:
            rollout_responses.append(out.text)
            repeated_ground_truths.append(gt)
            repeated_prompts.append(pr)

    return repeated_prompts, rollout_responses, repeated_ground_truths


@torch.no_grad()
def get_batched_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    micro_batch_size: int,
) -> torch.Tensor:
    """Compute log-probs in micro-batches to avoid OOM.

    Returns:
        torch.Tensor of shape (batch_size, seq_len) on the same device as input_ids.
    """
    model.eval()
    device = input_ids.device
    log_probs_list = []
    for start in range(0, input_ids.shape[0], micro_batch_size):
        end = start + micro_batch_size
        res = get_response_log_probs(
            model, input_ids[start:end], labels[start:end], return_token_entropy=False
        )
        log_probs_list.append(res["log_probs"].cpu())
    return torch.cat(log_probs_list, dim=0).to(device)


def grpo_train_loop(
    model_name_or_path: str = "models/Qwen2.5-Math-1.5B",
    train_data_path: str = "data/MATH/train.jsonl",
    val_data_path: str = "data/MATH/validation.jsonl",
    prompt_path: str = "cs336_alignment/prompts/r1_zero.prompt",
    output_dir: str = "output/grpo",
    save_best: bool = False,
    n_grpo_steps: int = 200,
    learning_rate: float = 4e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,  # On-policy
    train_batch_size: int = 256,  # On-policy
    gradient_accumulation_steps: int = 128,  # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.2,
    loss_type: Annotated[Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        "grpo_no_clip",
    ], typer.Option()] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    constant_normalize_factor: float | None = None,
    use_question_only_reward: bool = False,
    seed: int = 42,
    eval_steps: int = 10,
    clip_range: float = 0.2,  # DeepSeek's cliprange param for GRPO-Clip
    max_grad_norm: float = 1.0,
    run_name: str | None = None,
):
    random.seed(seed)
    torch.manual_seed(seed)

    # Select reward function based on flag
    reward_fn = question_only_reward_fn if use_question_only_reward else r1_zero_reward_fn
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    wandb.init(
        project="cs336-assignment5-alignment-grpo",
        name=run_name,
        config=locals(),
    )

    policy, tokenizer, vllm_engine = setup_hf_and_vllm(
        model_name_or_path=model_name_or_path,
        device=DEVICE,
        seed=seed,
        vllm_gpu_memory_utilization=gpu_memory_utilization,
        vllm_max_model_len=2048,
    )

    train_records = load_MATH(train_data_path, prompt_path, -1)
    val_records = load_MATH(val_data_path, prompt_path, 1024)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    rollout_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )

    global_update_step = 0
    best_acc = 0.0
    # Outer GRPO loop
    for step in range(1, n_grpo_steps + 1):
        logger.info(f"==== GRPO Step {step} ====")
        # Evaluate before generating except at the end
        if (step - 1) % eval_steps == 0:
            policy.eval()
            acc = evaluate_vllm_loop(policy, vllm_engine, val_records)
            wandb.log({"eval/accuracy": acc, "step": step})
            logger.info(f"Step {step} Accuracy: {acc:.4f}")
            if save_best and acc > best_acc:
                best_acc = acc
                save_path = os.path.join(output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                policy.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"Saved best model to {save_path} with accuracy {acc:.4f}")

        policy.eval()

        # Generate rollouts and flatten
        sampled_prompts = random.sample(train_records, n_prompts_per_rollout_batch)
        repeated_prompts, rollout_responses, repeated_ground_truths = (
            generate_and_flatten_rollouts(
                policy, vllm_engine, sampled_prompts, rollout_sampling_params
            )
        )

        # Compute rewards (reward_fn is called only once inside)
        rewards, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn,
            rollout_responses,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization,
        )
        rewards = rewards.to(DEVICE)
        raw_rewards = raw_rewards.to(DEVICE)

        # Tokenize
        batch_tensors = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )

        # Precompute old_log_probs for off-policy ratio-based losses (clip and no-clip)
        if loss_type in ("grpo_clip", "grpo_no_clip"):
            old_log_probs = get_batched_log_probs(
                policy,
                batch_tensors["input_ids"].to(DEVICE),
                batch_tensors["labels"].to(DEVICE),
                micro_train_batch_size,
            )
        else:
            old_log_probs = None

        # Log macro/rollout-level stats once per step
        wandb.log({
            "step": step,
            "train/avg_reward": reward_metadata["avg_total_reward"],
            "train/avg_format_reward": reward_metadata["avg_format_reward"],
            "train/avg_answer_reward": reward_metadata["avg_answer_reward"],
        })

        policy.train()

        # Inner epoch loop
        for epoch in range(epochs_per_rollout_batch):
            indices = torch.randperm(rollout_batch_size)

            for start_idx in range(0, rollout_batch_size, train_batch_size):
                chunk_indices = indices[start_idx : start_idx + train_batch_size]
                optimizer.zero_grad()

                accu_loss = 0.0
                accu_entropy = 0.0
                accu_clip_frac = 0.0
                actual_grad_accum_steps = max(
                    1, len(chunk_indices) // micro_train_batch_size
                )

                for micro_start in range(0, len(chunk_indices), micro_train_batch_size):
                    micro_indices = chunk_indices[
                        micro_start : micro_start + micro_train_batch_size
                    ]

                    mb_input_ids = batch_tensors["input_ids"][micro_indices].to(DEVICE)
                    mb_labels = batch_tensors["labels"][micro_indices].to(DEVICE)
                    mb_resp_mask = batch_tensors["response_mask"][micro_indices].to(DEVICE)
                    mb_rewards = rewards[micro_indices].unsqueeze(-1)
                    mb_raw_rewards = raw_rewards[micro_indices].unsqueeze(-1)
                    mb_old_log_probs = (
                        old_log_probs[micro_indices]
                        if old_log_probs is not None
                        else None
                    )

                    log_probs_output = get_response_log_probs(
                        policy, mb_input_ids, mb_labels, return_token_entropy=True
                    )

                    micro_loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=log_probs_output["log_probs"],
                        response_mask=mb_resp_mask,
                        gradient_accumulation_steps=actual_grad_accum_steps,
                        loss_type=loss_type,
                        raw_rewards=mb_raw_rewards,
                        advantages=mb_rewards,
                        old_log_probs=mb_old_log_probs,
                        cliprange=clip_range,
                        normalize_constant=constant_normalize_factor,
                    )

                    accu_loss += micro_loss.item()
                    accu_entropy += (
                        log_probs_output.get("token_entropy", torch.tensor(0.0))
                        .mean().item() / actual_grad_accum_steps
                    )

                    if "clip_frac" in metadata:
                        accu_clip_frac += (
                            metadata["clip_frac"].item() / actual_grad_accum_steps
                        )

                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                global_update_step += 1

                log_dict = {
                    "step": step,
                    "update_step": global_update_step,
                    "train/loss": accu_loss,
                    "train/token_entropy": accu_entropy,
                    "train/grad_norm": grad_norm.item()
                    if not isinstance(grad_norm, float)
                    else grad_norm,
                }
                if loss_type == "grpo_clip":
                    log_dict["train/clip_fraction"] = accu_clip_frac
                wandb.log(log_dict)

    # Final eval
    policy.eval()
    acc = evaluate_vllm_loop(policy, vllm_engine, val_records)
    wandb.log({"eval/accuracy": acc, "step": n_grpo_steps + 1})
    logger.info(f"Final Accuracy: {acc:.4f}")

    # Save checkpoint
    save_path = os.path.join(output_dir, "final_grpo_model")
    os.makedirs(save_path, exist_ok=True)
    policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved final model to {save_path}")
    wandb.finish()

    # destroy_process_group
    if dist.is_initialized():  # ty:ignore[possibly-missing-attribute]
        dist.destroy_process_group()  # ty:ignore[possibly-missing-attribute]


if __name__ == "__main__":
    typer.run(grpo_train_loop)
