import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    concat_ids_batch = []
    input_ids_batch = []
    labels_batch = []
    mask_batch = []
    max_len = 0

    for prompt, output in zip(prompt_strs, output_strs):
        # tokenizes the question and output separately
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        # concatenates them together
        concat_ids = prompt_ids + output_ids
        concat_ids_batch.append(concat_ids)

        # prompt part in labels has length max(0, p_len - 1)
        mask = [0] * max(0, len(prompt_ids) - 1) + [1] * len(output_ids)
        mask_batch.append(mask)

        max_len = max(max_len, len(concat_ids))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for i in range(len(concat_ids_batch)):
        concat_ids = concat_ids_batch[i]
        pad_len = max_len - len(concat_ids)
        padded = concat_ids + [pad_id] * pad_len

        # the tokenized prompt and output strings, with the final token sliced off
        input_ids = padded[:-1]
        input_ids_batch.append(input_ids)
        # shifted input ids, i.e., the input ids without the first token
        labels = padded[1:]
        labels_batch.append(labels)

        mask_batch[i].extend([0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "response_mask": torch.tensor(mask_batch, dtype=torch.bool),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor
            Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.
    Returns:
        torch.Tensor
            Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """
    # $$ H(p) = \text{LogSumExp}(z) - \sum_{i} p_i z_i $$
    p = torch.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(p * logits, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    log_p = torch.log_softmax(logits, dim=-1)
    # Gather the log probabilities of the actual label tokens
    log_probs = torch.gather(log_p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result
