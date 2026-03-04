from pathlib import Path

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import evaluate_vllm, load_MATH

if __name__ == "__main__":
    data_path = "data/MATH/validation.jsonl"
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    records = load_MATH(data_path, prompt_path)
    examples = {r["prompt"]: r["ground_truth"] for r in records}

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True
    )
    llm = LLM(model="models/Qwen2.5-Math-1.5B")
    output_path = Path("data/MATH/output/math_baseline.jsonl")
    evaluate_vllm(llm, r1_zero_reward_fn, examples, sampling_params, output_path)
