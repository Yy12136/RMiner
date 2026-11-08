"""Discover simulator task profiles from the task launch/collection scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ASSEMBLE_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = ASSEMBLE_ROOT / "task"


@dataclass(frozen=True)
class TaskProfile:
    name: str
    instruction: str
    task_id: str
    sim_launcher: Path
    effect_center: str
    reset_mode: str


# Only tasks with a calibrated non-origin work area need an override. Other
# task scenes use automatic camera-based centering inside Isaac Sim.
EFFECT_CENTER_OVERRIDES = {
    "g1_tabletop_grasp_task": "-3.0606229131,3.5318245338,0",
}


def _extract_instruction(text: str) -> str | None:
    match = re.search(
        r'^TASK_INSTRUCTION="\$\{G1_TASK_INSTRUCTION:-(.*)\}"$',
        text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _extract_task_id(text: str) -> str | None:
    match = re.search(r"--task\s+['\"]?([A-Za-z0-9_-]+)", text)
    if match and match.group(1) != "$TASK_ID":
        return match.group(1)
    match = re.search(r'TASK_ID="\$\{[^:}]+:-([A-Za-z0-9_-]+)\}"', text)
    return match.group(1) if match else None


def discover_profiles() -> list[TaskProfile]:
    profiles = []
    for directory in sorted(TASKS_ROOT.glob("g1_*_task")):
        quest = directory / "run_quest3s.sh"
        simulator = directory / "run_task.sh"
        if not quest.is_file() or not simulator.is_file():
            continue
        quest_text = quest.read_text(encoding="utf-8", errors="replace")
        simulator_text = simulator.read_text(encoding="utf-8", errors="replace")
        instruction = _extract_instruction(quest_text)
        task_id = _extract_task_id(simulator_text)
        if not instruction or not task_id:
            continue
        reset_mode = (
            "ack"
            if "G1_BENCHMARK_RESET_STATUS_FILE" in simulator_text
            else "dds"
        )
        profiles.append(
            TaskProfile(
                name=directory.name,
                instruction=instruction,
                task_id=task_id,
                sim_launcher=simulator.resolve(),
                effect_center=EFFECT_CENTER_OVERRIDES.get(directory.name, "auto"),
                reset_mode=reset_mode,
            )
        )
    return profiles


def resolve_profile(instruction: str) -> TaskProfile:
    matches = [profile for profile in discover_profiles() if profile.instruction == instruction]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = "\n".join(
            f"  - {profile.name}: {profile.instruction}"
            for profile in discover_profiles()
        )
        raise ValueError(
            "No installed task profile matches the dataset instruction. "
            "Provide --task-id, --sim-launcher, --effect-center and --reset-mode, "
            "or add a run_quest3s.sh/run_task.sh pair under assemble/task.\n"
            f"Discovered profiles:\n{available}"
        )
    raise ValueError(
        "More than one installed task uses this instruction; provide explicit "
        "--task-id and --sim-launcher."
    )
