"""Run selected few-shot training launchers."""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingTask:
    model: str
    shot: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pix2pix", "cyclegan"],
        default=["pix2pix", "cyclegan"],
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="List of shot counts",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="List of seeds",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        help=(
            "Run independent model/shot/seed jobs concurrently, with at most "
            "one job on each listed physical GPU"
        ),
    )
    return parser.parse_args()


def build_tasks(models: list[str], shots: list[int], seeds: list[int]) -> list[TrainingTask]:
    return [
        TrainingTask(model=model, shot=shot, seed=seed)
        for model in models
        for shot in shots
        for seed in seeds
    ]


def build_launcher_command(
    model: str,
    shots: list[int],
    seeds: list[int],
    config: Path,
    gpu: int | None = None,
) -> list[str]:
    scripts = {
        "pix2pix": PROJECT_ROOT / "src" / "train" / "model_paired.py",
        "cyclegan": PROJECT_ROOT / "src" / "train" / "model_unpaired.py",
    }
    command = [
        sys.executable,
        str(scripts[model]),
        "--shots",
        *map(str, shots),
        "--seeds",
        *map(str, seeds),
        "--config",
        str(config.resolve()),
    ]
    if gpu is not None:
        command.extend(["--gpu", str(gpu)])
    return command


def run_command(command: list[str], gpu: int | None = None) -> None:
    prefix = f"[GPU {gpu}]" if gpu is not None else "[CMD]"
    print(prefix, " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_parallel(tasks: list[TrainingTask], gpus: list[int], config: Path) -> None:
    pending: queue.Queue[TrainingTask] = queue.Queue()
    for task in tasks:
        pending.put(task)

    stop = threading.Event()
    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while not stop.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            try:
                command = build_launcher_command(
                    task.model, [task.shot], [task.seed], config, gpu
                )
                run_command(command, gpu)
            except BaseException as error:
                with error_lock:
                    errors.append(error)
                stop.set()
            finally:
                pending.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()

    if errors:
        raise RuntimeError("Parallel training stopped after a worker failed") from errors[0]


def main() -> int:
    args = parse_args()
    config = args.config.resolve()
    if args.gpus is not None:
        if any(gpu < 0 for gpu in args.gpus):
            raise ValueError("GPU indices must be non-negative")
        if len(set(args.gpus)) != len(args.gpus):
            raise ValueError("GPU indices must be unique")
        run_parallel(build_tasks(args.models, args.shots, args.seeds), args.gpus, config)
    else:
        for model in args.models:
            run_command(
                build_launcher_command(model, args.shots, args.seeds, config)
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
