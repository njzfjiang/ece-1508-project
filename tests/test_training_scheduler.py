import threading
import time
from pathlib import Path

from scripts import train_models


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_build_tasks_expands_the_model_shot_seed_grid():
    tasks = train_models.build_tasks(["cyclegan", "pix2pix"], [10, 20], [1, 2])

    assert len(tasks) == 8
    assert tasks[0] == train_models.TrainingTask("cyclegan", 10, 1)
    assert tasks[-1] == train_models.TrainingTask("pix2pix", 20, 2)


def test_launcher_command_assigns_exactly_one_run_to_a_gpu(tmp_path):
    command = train_models.build_launcher_command(
        "pix2pix", [20], [3], Path(tmp_path / "config.yaml"), gpu=2
    )

    assert _value_after(command, "--shots") == "20"
    assert _value_after(command, "--seeds") == "3"
    assert _value_after(command, "--gpu") == "2"
    assert command[1].endswith("model_paired.py")


def test_parallel_scheduler_runs_each_task_once_and_serializes_each_gpu(monkeypatch):
    tasks = train_models.build_tasks(["cyclegan"], [10, 20], [1, 2])
    active: set[int] = set()
    seen: list[tuple[int, int, int]] = []
    lock = threading.Lock()

    def fake_run(command: list[str], gpu: int | None = None) -> None:
        assert gpu is not None
        with lock:
            assert gpu not in active
            active.add(gpu)
            seen.append(
                (
                    gpu,
                    int(_value_after(command, "--shots")),
                    int(_value_after(command, "--seeds")),
                )
            )
        time.sleep(0.01)
        with lock:
            active.remove(gpu)

    monkeypatch.setattr(train_models, "run_command", fake_run)
    train_models.run_parallel(tasks, [0, 1], Path("configs/base.yaml"))

    assert len(seen) == len(tasks)
    assert {(shot, seed) for _, shot, seed in seen} == {
        (10, 1),
        (10, 2),
        (20, 1),
        (20, 2),
    }
