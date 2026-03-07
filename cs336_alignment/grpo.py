from typing import Callable, Literal

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    return (rewards.flatten(), raw_rewards, {})


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

    return (-torch.min(part1, part2), {})


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
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
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
    if loss_type not in ("no_baseline", "reinforce_with_baseline", "grpo_clip"):
        raise NotImplementedError

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type,
        raw_rewards, advantages, old_log_probs, cliprange,  # ty:ignore[invalid-argument-type]
    )
    batch_loss = masked_mean(per_token_loss, response_mask, dim=-1)

    # mean loss across sequence
    actual_loss = batch_loss.mean()
    loss = actual_loss / gradient_accumulation_steps
    # autograd
    loss.backward()

    metadata |= {"policy_loss": actual_loss.detach()}
    return loss, metadata
