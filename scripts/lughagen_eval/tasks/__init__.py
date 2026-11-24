from __future__ import annotations

from .base import SCORING_MODES, Task
from .belebele import BelebeleTask
from .flores import FloresTask
from .irokobench import AfriMgsmTask, AfriMmluTask, AfriXnliTask
from .sib import SibTask

# Tasks that take a language directly (one task instance handles every language).
SINGLE_INSTANCE_TASKS: dict[str, type[Task]] = {
    "sib": SibTask,
    "belebele": BelebeleTask,
    "afrimmlu": AfriMmluTask,
    "afrixnli": AfriXnliTask,
    "afrimgsm": AfriMgsmTask,
}

# FLORES is special: one task *instance* per translation direction, since the
# source/target codes are fixed at construction rather than passed per call.
DIRECTIONAL_TASKS = {"flores": FloresTask}

ALL_TASK_NAMES = sorted(set(SINGLE_INSTANCE_TASKS) | set(DIRECTIONAL_TASKS))


def get_task(name: str) -> Task:
    if name not in SINGLE_INSTANCE_TASKS:
        raise KeyError(
            f"{name!r} is not a single-instance task; construct FloresTask directly "
            f"with a source/target language pair instead."
        )
    return SINGLE_INSTANCE_TASKS[name]()


__all__ = [
    "Task", "SCORING_MODES", "SibTask", "FloresTask", "BelebeleTask",
    "AfriMmluTask", "AfriXnliTask", "AfriMgsmTask",
    "SINGLE_INSTANCE_TASKS", "DIRECTIONAL_TASKS", "ALL_TASK_NAMES", "get_task",
]
