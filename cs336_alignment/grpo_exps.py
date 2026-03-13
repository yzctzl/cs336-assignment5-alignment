import logging
import subprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def grpo_learning_rate(lrs: list[float]):
    for lr in lrs:
        run_name = f"grpo_lr_{lr:.0e}"
        output_dir = f"output/grpo_{run_name}"

        cmd = [
            "uv",
            "run",
            "cs336_alignment/grpo.py",
            f"--learning-rate={lr}",
            f"--run-name={run_name}",
            f"--output-dir={output_dir}",
        ]

        logger.info(f"Sweep: {run_name}\nCmd: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error: Experiment {run_name} failed with return code {e.returncode}"
            )
        except KeyboardInterrupt:
            logger.warning("\nSweep interrupted by user.")
            continue

def grpo_baselines(loss_types: list[str]):
    for loss_type in loss_types:
        run_name = f"grpo_loss_type_{loss_type}"
        output_dir = f"output/grpo_{run_name}"

        cmd = [
            "uv",
            "run",
            "cs336_alignment/grpo.py",
            f"--loss-type={loss_type}",
            f"--run-name={run_name}",
            f"--output-dir={output_dir}",
        ]

        logger.info(f"Sweep: {run_name}\nCmd: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error: Experiment {run_name} failed with return code {e.returncode}"
            )
        except KeyboardInterrupt:
            logger.warning("\nSweep interrupted by user.")
            continue

def grpo_length_normalization(factors: list[float | None]):
    for factor in factors:
        name = "masked_mean" if factor is None else f"constant_{factor}"
        run_name = f"grpo_norm_{name}"
        output_dir = f"output/grpo_{run_name}"

        cmd = [
            "uv",
            "run",
            "cs336_alignment/grpo.py",
            f"--run-name={run_name}",
            f"--output-dir={output_dir}",
        ]
        if factor is not None:
            cmd.append(f"--constant-normalize-factor={factor}")

        logger.info(f"Sweep: {run_name}\nCmd: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error: Experiment {run_name} failed with return code {e.returncode}"
            )
        except KeyboardInterrupt:
            logger.warning("\nSweep interrupted by user.")
            continue

def grpo_group_standard_deviation(use_group_stds: list[bool]):
    for use_std in use_group_stds:
        run_name = f"grpo_group_std_{str(use_std).lower()}"
        output_dir = f"output/grpo_{run_name}"

        cmd = [
            "uv",
            "run",
            "cs336_alignment/grpo.py",
            "--use-std-normalization" if use_std else "--no-use-std-normalization",
            f"--run-name={run_name}",
            f"--output-dir={output_dir}",
        ]

        logger.info(f"Sweep: {run_name}\nCmd: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error: Experiment {run_name} failed with return code {e.returncode}"
            )
        except KeyboardInterrupt:
            logger.warning("\nSweep interrupted by user.")
            continue


def grpo_off_policy(
    epochs_per_rollout: int, 
    train_batch_size: int, 
    n_steps: int = 200,
    run_name_extra: str = ""
):
    """
    Runs an off-policy experiment with GRPO-Clip.
    To keep memory constant (microbatch = 2), grad_accum_steps = train_batch_size // 2.
    """
    loss_type = "grpo_clip"
    grad_accum_steps = train_batch_size // 2
    
    run_name = f"grpo_off_policy_ep{epochs_per_rollout}_bs{train_batch_size}{run_name_extra}"
    output_dir = f"output/grpo_{run_name}"

    cmd = [
        "uv",
        "run",
        "cs336_alignment/grpo.py",
        f"--loss-type={loss_type}",
        f"--epochs-per-rollout-batch={epochs_per_rollout}",
        f"--train-batch-size={train_batch_size}",
        f"--gradient-accumulation-steps={grad_accum_steps}",
        f"--n-grpo-steps={n_steps}",
        f"--run-name={run_name}",
        f"--output-dir={output_dir}",
    ]

    logger.info(f"Off-policy Experiment: {run_name}\nCmd: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: Experiment {run_name} failed with code {e.returncode}")
    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted.")


def grpo_off_policy_sweep(epochs_list: list[int], bs_list: list[int], n_steps: int = 50):
    """
    Sweeps combinations of epochs and batch sizes for off-policy GRPO.
    """
    for ep in epochs_list:
        for bs in bs_list:
            grpo_off_policy(ep, bs, n_steps=n_steps, run_name_extra="_sweep")


def grpo_off_policy_clip_ablation(
    epochs_per_rollout: int, 
    train_batch_size: int, 
    n_steps: int = 200,
):
    """
    Ablates GRPO-Clip by using 'grpo_no_clip' (ratio * advantage without clipping).
    Use your best off-policy hyperparameters from grpo_off_policy_sweep.
    """
    loss_type = "grpo_no_clip"
    grad_accum_steps = train_batch_size // 2

    run_name = f"grpo_no_clip_ep{epochs_per_rollout}_bs{train_batch_size}"
    output_dir = f"output/grpo_{run_name}"

    cmd = [
        "uv",
        "run",
        "cs336_alignment/grpo.py",
        f"--loss-type={loss_type}",
        f"--epochs-per-rollout-batch={epochs_per_rollout}",
        f"--train-batch-size={train_batch_size}",
        f"--gradient-accumulation-steps={grad_accum_steps}",
        f"--n-grpo-steps={n_steps}",
        f"--run-name={run_name}",
        f"--output-dir={output_dir}",
    ]

    logger.info(f"Clip Ablation: {run_name}\nCmd: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: Experiment {run_name} failed with code {e.returncode}")
    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted.")


def grpo_prompt_ablation(n_steps: int = 200):
    """
    Compares:
    1. r1_zero prompt + r1_zero_reward_fn (default)
    2. question_only prompt + question_only_reward_fn

    Note: The prompt affects the reward function used both during training and validation.
    See drgrpo_grader.py for question_only_reward_fn.
    """
    configs = [
        {
            "prompt_path": "cs336_alignment/prompts/r1_zero.prompt",
            "name": "r1_zero_prompt",
            # default reward fn (r1_zero) is used when no --reward-fn flag is provided
        },
        {
            "prompt_path": "cs336_alignment/prompts/question_only.prompt",
            "name": "question_only_prompt",
            "--use-question-only-reward": True,
        },
    ]

    for config in configs:
        run_name = f"grpo_prompt_ablation_{config['name']}"
        output_dir = f"output/grpo_{run_name}"

        cmd = [
            "uv",
            "run",
            "cs336_alignment/grpo.py",
            f"--prompt-path={config['prompt_path']}",
            f"--n-grpo-steps={n_steps}",
            f"--run-name={run_name}",
            f"--output-dir={output_dir}",
        ]
        if config.get("--use-question-only-reward"):
            cmd.append("--use-question-only-reward")

        logger.info(f"Prompt Ablation: {run_name}\nCmd: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error: Experiment {run_name} failed with code {e.returncode}")
        except KeyboardInterrupt:
            logger.warning("\nExperiment interrupted.")


if __name__ == "__main__":
    # grpo_learning_rate([5e-6, 1e-5, 2e-5, 3e-5, 4e-5])
    # grpo_baselines(["no_baseline", "reinforce_with_baseline"])
    grpo_length_normalization([1024.0])
    grpo_group_standard_deviation([False])

    # Off-policy experiments
    # Broad sweep (< 50 steps, pick best combo)
    grpo_off_policy_sweep(epochs_list=[2, 4], bs_list=[128, 256], n_steps=40)

    # Focused run with best combo (200 steps), also run on-policy baseline:
    grpo_off_policy(epochs_per_rollout=1, train_batch_size=256, n_steps=200)  # on-policy baseline
    grpo_off_policy(epochs_per_rollout=2, train_batch_size=256, n_steps=200)  # example off-policy

    # Clip ablation (use your best off-policy hyperparams)
    grpo_off_policy_clip_ablation(epochs_per_rollout=2, train_batch_size=256, n_steps=200)

    # Prompt ablation
    grpo_prompt_ablation(n_steps=200)
    pass
