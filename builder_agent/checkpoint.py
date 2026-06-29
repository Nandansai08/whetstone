from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

from builder_agent import config
from builder_agent.schemas import Plan, Spec, SubTask


def build_id(request: str) -> str:
    return hashlib.sha256(request.encode()).hexdigest()[:16]


def _path(bid: str) -> str:
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(config.CHECKPOINT_DIR, f"{bid}.json")


def save(
    bid: str,
    spec: Spec,
    plan: Plan,
    outputs: dict[str, str],
    completed_ids: set[str],
) -> None:
    data = {
        "spec": asdict(spec),
        "plan": asdict(plan),
        "outputs": outputs,
        "completed_ids": sorted(completed_ids),
    }
    path = _path(bid)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def load(bid: str) -> dict | None:
    path = _path(bid)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "spec": Spec(**data["spec"]),
        "plan": Plan(subtasks=[SubTask(**s) for s in data["plan"]["subtasks"]]),
        "outputs": data["outputs"],
        "completed_ids": set(data["completed_ids"]),
    }


def clear(bid: str) -> None:
    path = _path(bid)
    if os.path.exists(path):
        os.remove(path)
