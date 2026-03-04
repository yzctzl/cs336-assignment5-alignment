import argparse
import json
import logging
import os
from unittest.mock import patch

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    evaluate_vllm,
    get_response_log_probs,
    load_MATH,
    masked_normalize,
    tokenize_prompt_and_output,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_vllm(
    model_id: str,
    device: str,
    seed: int = 42,
    gpu_memory_utilization: float = 0.35,
    max_model_len: int = 2048,
):
    """
    Start the inference process, using vLLM to hold a model.
    Includes memory constraints to comfortably coexist with PyTorch on a single A100.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch to ensure safe startup on single-GPU without DDP interference
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        logger.info(
            f"Initializing vLLM on {device} with memory util={gpu_memory_utilization}"
        )
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,  # limit max model len to reduce VRAM
            enforce_eager=True,  # Disables CUDA graphs to save ~1-2GB of VRAM
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Sync weights from the PyTorch SFT policy directly into the vLLM engine instance.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model  # ty:ignore[unresolved-attribute]
    llm_model.load_weights(state_dict.items())


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (batch_size, sequence_length)
            per-token log-probabilities from the SFT policy being trained.
        response_mask (batch_size, sequence_length)
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps
            Number of microbatches per optimizer step.
        normalize_constant
            The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns: tuple[torch.Tensor, dict[str, torch.Tensor]]
        loss scalar tensor
            The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
        metadata Dict with metadata from the underlying loss call,
            and any other statistics you might want to log.
    """
    # maximize the log-probabilities of correct tokens is minimize negative policy_log_probs
    per_token_loss = -policy_log_probs

    # masked_normalize
    batch_loss = masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1,  # loss of a sequence
    )

    # mean loss across sequence
    scaled_loss = batch_loss.mean() / gradient_accumulation_steps
    # autograd
    scaled_loss.backward()

    metadata = {"loss": batch_loss.detach().mean()}
    return scaled_loss, metadata


def collate_fn(batch, tokenizer):
    prompts = [item["prompt"] for item in batch]
    outputs = [item.get("response", "") for item in batch]
    return tokenize_prompt_and_output(prompts, outputs, tokenizer)


def evaluate_vllm_loop(
    model: PreTrainedModel,
    llm: LLM,
    val_records: list[dict[str, str]],
):
    """
    Evaluate the model using vLLM for drastically faster generation inside the training process.
    """
    # Sync weights to vLLM
    load_policy_into_vllm_instance(model, llm)
    logger.info(f"Running vLLM evaluation on {len(val_records)} examples...")

    examples = {
        r["prompt"]: str(r.get("response") or r.get("ground_truth", ""))
        for r in val_records
    }
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    accuracy, format_acc, _ = evaluate_vllm(
        llm, r1_zero_reward_fn, examples, sampling_params, None
    )
    logger.info(
        f"Format Reward: {format_acc:.4f} | Answer Reward (Acc): {accuracy:.4f}"
    )

    # Empty pyTorch cache out of an abundance of caution due to vLLM execution
    torch.cuda.empty_cache()
    return accuracy


def filter_sft_data(input_path, output_path):
    correct_count = 0
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            reward = r1_zero_reward_fn(item["response"], item["ground_truth"])
            if reward["reward"] > 0:
                f_out.write(json.dumps(item) + "\n")
                correct_count += 1
    logger.info(f"Filter the reasoning SFT examples, keep {correct_count}.")


def sft_train_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_records: list[dict],
    *,
    device: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    epochs: int,
    warmup_steps: int,
    max_grad_norm: float,
    start_step: int = 0,
    eval_steps: int | None = None,
    vllm_engine: LLM | None = None,
    val_records: list[dict[str, str]] | None = None,
    output_dir: str | None = None,
):
    """
    Reusable SFT training loop. Used by both standalone SFT and Expert Iteration.

    If vllm_engine / val_records / output_dir are provided, periodic eval and
    checkpointing are performed every `eval_steps` gradient updates.
    """
    train_loader = DataLoader(
        train_records,  # ty:ignore[invalid-argument-type]
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    total_steps = (
        len(train_loader) * epochs + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    if total_steps == 0:
        logger.warning("Too few examples for a full gradient step — skipping.")
        return 0

    logger.info(f"Starting SFT for {total_steps} gradient steps ({epochs} epochs).")

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    optimizer.zero_grad()

    train_step = start_step
    eval_step = 0
    accumulated_loss = 0.0
    accumulated_entropy = 0.0
    accumulate_count = 0

    model.train()
    for epoch in range(epochs):
        for idx, batch_tensors in enumerate(train_loader):
            input_ids = batch_tensors["input_ids"].to(device)
            labels = batch_tensors["labels"].to(device)
            response_mask = batch_tensors["response_mask"].to(device)

            log_probs_output = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )

            _, metadata = sft_microbatch_train_step(
                policy_log_probs=log_probs_output["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=response_mask.sum(dim=-1),
            )

            accumulated_loss += metadata["loss"].item() / gradient_accumulation_steps
            accumulated_entropy += (
                log_probs_output.get("token_entropy", torch.tensor(0.0)).mean().item()
                / gradient_accumulation_steps
            )

            accumulate_count += 1
            is_last_step = (epoch == epochs - 1) and (idx == len(train_loader) - 1)

            if accumulate_count == gradient_accumulation_steps or is_last_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_step += 1
                accumulate_count = 0

                wandb.log(
                    {
                        "train_step": train_step,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/loss": accumulated_loss,
                        "train/token_entropy": accumulated_entropy,
                        "train/grad_norm": grad_norm.item()
                        if not isinstance(grad_norm, float)
                        else grad_norm,
                    }
                )
                logger.info(
                    f"Epoch {epoch} | Step {train_step} | Loss: {accumulated_loss:.4f}"
                )
                accumulated_loss = 0.0
                accumulated_entropy = 0.0

                # Periodic evaluation & checkpoint (optional)
                if (
                    vllm_engine is not None
                    and val_records is not None
                    and eval_steps
                    and (train_step % eval_steps == 0 or train_step == total_steps)
                ):
                    eval_step += 1
                    model.eval()
                    accuracy = evaluate_vllm_loop(model, vllm_engine, val_records)
                    wandb.log({"eval_step": eval_step, "eval/accuracy": accuracy})
                    if output_dir:
                        save_path = os.path.join(output_dir, f"checkpoint-{train_step}")
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                    model.train()

    torch.cuda.empty_cache()
    return train_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/Qwen2.5-Math-1.5B"
    )
    parser.add_argument("--sft_data_path", type=str, default="data/MATH/sft.jsonl")
    parser.add_argument(
        "--val_data_path", type=str, default="data/MATH/validation.jsonl"
    )
    parser.add_argument("--output_dir", type=str, default="output/sft")

    # Hyperparameters
    parser.add_argument("--dataset_size", type=int, default=-1)
    parser.add_argument(
        "--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--num_eval_batches", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=6)
    parser.add_argument("--run_name", type=str, default="sft_run")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init WandB
    wandb.init(
        project="cs336-assignment5-alignment-sft", name=args.run_name, config=vars(args)
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Load Model & Tokenizer (PyTorch Training Pool)
    logger.info(f"Loading {args.model_name_or_path} for PyTorch ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    # Init vLLM (vLLM Evaluation Pool - Restricting to 0.35)
    vllm_engine = init_vllm(
        args.model_name_or_path,
        device=str(device),
        gpu_memory_utilization=0.35,
        max_model_len=2048,
    )

    # Load Data
    train_records = load_MATH(args.sft_data_path, args.prompt_path, args.dataset_size)
    val_records = load_MATH(
        args.val_data_path,
        args.prompt_path,
        min(512, args.num_eval_batches * args.batch_size * 4),
    )

    sft_train_loop(
        model=model,
        tokenizer=tokenizer,
        train_records=train_records,
        device=str(device),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        eval_steps=args.eval_steps,
        vllm_engine=vllm_engine,
        val_records=val_records,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    # filter_sft_data("data/MATH/sft.jsonl", "data/MATH/filtered_sft.jsonl")
    main()
