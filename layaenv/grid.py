from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
except Exception:  # pyyaml missing
    yaml = None

GRID_SCHEMA = "1.0"


@dataclass
class Cell:
    id: str
    domain: str
    split: str
    question_types: list
    models: list
    source: str = "hf"
    hf_repo: str = ""
    hf_config: str = ""
    hf_split: str = "test"
    processor: str = ""
    local_file: str = ""
    max_instances: Optional[int] = None
    sample_seed: int = 0
    notes: list = field(default_factory=list)


def load_grid(path: str) -> tuple[str, list[Cell]]:
    if yaml is None:
        raise RuntimeError("pyyaml required to parse the grid config")
    with open(path) as f:
        raw = yaml.safe_load(f)
    ver = str(raw.get("schema_version", ""))
    if ver != GRID_SCHEMA:
        raise ValueError(f"grid schema mismatch: {ver} != {GRID_SCHEMA}")
    tag = str(raw.get("grid_tag", "untagged"))
    cells = []
    for c in raw.get("cells", []):
        cells.append(
            Cell(
                id=c["id"],
                domain=str(c.get("domain", "")),
                split=str(c.get("split", "id")),
                question_types=list(c.get("question_types", ["choice", "score", "noul"])),
                models=list(c.get("models", [])),
                source=str(c.get("source", "hf")),
                hf_repo=str(c.get("dataset", {}).get("hf", {}).get("repo", "")),
                hf_config=str(c.get("dataset", {}).get("hf", {}).get("config", "")),
                hf_split=str(c.get("dataset", {}).get("hf", {}).get("split", "test")),
                processor=str(c.get("processor", "")),
                local_file=str(c.get("dataset", {}).get("local", {}).get("file", "")),
                max_instances=c.get("max_instances"),
                sample_seed=int(c.get("sample_seed", 0)),
                notes=list(c.get("notes", [])),
            )
        )
    return tag, cells


def resolve_local_path(cell: Cell, repo_root: str) -> str:
    return os.path.join(repo_root, cell.local_file)