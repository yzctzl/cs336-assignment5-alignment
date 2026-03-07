import argparse
import logging
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import (
    evaluate_vllm_loop,
    init_vllm,
    load_policy_into_vllm_instance,
    sft_train_loop,
)
from cs336_alignment.utils import load_MATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# generate rollouts & filter correct ones
def generate_and_filter(model, llm, questions, sampling_params):
    """
    For each question, generate candidate answers via the
    current policy, then keep only those judged correct by the reward function.

    Returns list[dict] with keys {"prompt", "response"} ready for SFT.
    """
    load_policy_into_vllm_instance(model, llm)

    prompts, ground_truths = [], []
    for q in questions:
        prompts.append(q["prompt"])
        ground_truths.append(str(q.get("ground_truth", "")))

    logger.info(f"Generating rollouts for {len(questions)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    # Keep only correct outputs
    sft_records = []
    total_rollouts = 0
    for output, gt in zip(outputs, ground_truths):
        for out in output.outputs:
            total_rollouts += 1
            response = out.text
            # min_tokens=4 cause bug, checked here
            if r1_zero_reward_fn(response, gt)["reward"] > 0:
                sft_records.append({"prompt": output.prompt, "response": response})

    logger.info(
        f"Filtered: {len(sft_records)} / {total_rollouts} correct "
        f"({len(sft_records) / max(total_rollouts, 1) * 100:.1f}%)"
    )
    return sft_records


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration")
    # Paths
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/Qwen2.5-Math-1.5B"
    )
    parser.add_argument("--train_data_path", type=str, default="data/MATH/train.jsonl")
    parser.add_argument(
        "--val_data_path", type=str, default="data/MATH/validation.jsonl"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt"
    )
    parser.add_argument("--output_dir", type=str, default="output/ei")

    # EI outer loop
    parser.add_argument(
        "--n_ei_steps", type=int, default=5, help="Number of EI iterations"
    )
    parser.add_argument(
        "--ei_batch_size", type=int, default=-1, help="Questions sampled per EI step"
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=4, help="Rollouts per question (G)"
    )

    # SFT inner loop (passed through to sft_train_loop)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--sft_epochs", type=int, default=2, help="SFT epochs per EI step"
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Eval & logging
    parser.add_argument("--num_eval_examples", type=int, default=500)
    parser.add_argument("--run_name", type=str, default="ei_run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # WandB
    wandb.init(
        project="cs336-assignment5-alignment-ei", name=args.run_name, config=vars(args)
    )

    # Load model & tokenizer
    logger.info(f"Loading model from {args.model_name_or_path} ...")
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

    # Init vLLM
    vllm_engine = init_vllm(args.model_name_or_path, device=str(device))

    # Load data
    train_records = load_MATH(args.train_data_path, args.prompt_path, -1)
    val_records = load_MATH(
        args.val_data_path, args.prompt_path, args.num_eval_examples
    )

    rollout_sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        # min_tokens=4,
        n=args.num_rollouts,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    global_train_step = 0
    # Expert Iteration outer loop
    for ei_step in range(1, args.n_ei_steps + 1):
        logger.info(f"{'=' * 60}")
        logger.info(f"EI Step {ei_step} / {args.n_ei_steps}")
        logger.info(f"{'=' * 60}")

        # sample a batch of questions
        questions = random.sample(
            train_records, min(args.ei_batch_size, len(train_records))
        )

        # generate rollouts & filter correct ones
        sft_records = generate_and_filter(
            model,
            vllm_engine,
            questions,
            sampling_params=rollout_sampling_params,
        )

        grad_accum_steps = args.gradient_accumulation_steps
        if len(sft_records) > 0:
            max_possible_accum = max(1, len(sft_records) // args.batch_size)
            grad_accum_steps = min(grad_accum_steps, max_possible_accum)

        # SFT on filtered data (reuses the exact same loop as sft.py)
        sft_steps = sft_train_loop(
            model=model,
            tokenizer=tokenizer,
            train_records=sft_records,
            device=str(device),
            batch_size=args.batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=args.learning_rate,
            epochs=args.sft_epochs,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            start_step=global_train_step,
            # No periodic eval inside inner loop; we eval after each EI step
        )
        global_train_step = sft_steps
        logger.info(
            f"EI Step {ei_step}: completed SFT up to global step {global_train_step}."
        )

        # Evaluate after this EI step
        model.eval()
        acc = evaluate_vllm_loop(model, vllm_engine, val_records)
        wandb.log({"eval/accuracy": acc, "ei_step": ei_step})
        logger.info(f"EI Step {ei_step}: accuracy = {acc:.4f}")

        # Save checkpoint
        save_path = os.path.join(args.output_dir, f"ei-step-{ei_step}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    wandb.finish()
    logger.info("Expert Iteration complete.")


if __name__ == "__main__":
    main()
