import json
import os
from pathlib import Path
from typing import Callable

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def prompt_setup(data_path: os.PathLike, prompt_path: os.PathLike):
    """
    load the examples and prompt the language model to 
    answer the question using the r1_zero prompt
    """
    examples: dict[str, str] = {}
    with open(prompt_path, encoding="utf-8") as prompt_file:
        prompt = prompt_file.read()

    with open(data_path) as jsonl:
        for line in jsonl:
            data = json.loads(line)
            prompted = prompt.format(question=data["problem"])
            # examples.append({
            #     "prompt": prompted,
            #     "level":data["level"],
            #     "type":data["type"],
            #     "solution":data["solution"]
            # })
            examples[prompted] = data["solution"]

    return examples


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    examples: dict[str, str],  # {prompt: answer} pair
    eval_sampling_params: SamplingParams,
    output_path: os.PathLike
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    prompts = list(examples.keys())
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # compute evaluation metrics
    format_reward = 0
    answer_reward = 0
    final_reward = 0
    for output in outputs:
        prompt = output.prompt
        answer = examples[prompt]  # ty:ignore[invalid-argument-type]
        response = output.outputs[0].text
        reward = reward_fn(response, answer)
        results.append({
            "reward": reward,
            "prompt": prompt,
            "answer": answer,
            "generated": response
        })

        format_reward += reward["format_reward"]
        answer_reward += reward["answer_reward"]
        final_reward += reward["reward"]

    print(f"format reward: {format_reward}\nanswer reward: {answer_reward}\nreward: {final_reward}")

    # serialize results to disk
    with open(output_path, "w") as fout:
        for result in results:
            fout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    data_path = Path("data/MATH/validation.jsonl")
    prompt_path = Path("cs336_alignment/prompts/r1_zero.prompt")
    examples = prompt_setup(data_path, prompt_path)

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True
    )
    llm = LLM(model="models/Qwen2.5-Math-1.5B")
    output_path = Path("data/MATH/output/math_baseline.jsonl")
    evaluate_vllm(llm, r1_zero_reward_fn, examples, sampling_params, output_path)
