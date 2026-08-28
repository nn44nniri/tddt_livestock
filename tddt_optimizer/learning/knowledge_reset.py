from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
from typing import Iterable

from ..database.sqlite_store import SQLiteStore


@dataclass
class ResetResult:
    reset_scope: str
    dry_run: bool
    removed_paths: list[str]
    skipped_paths: list[str]
    recreated_database: str | None
    preserved_prepared_knowledge: bool
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _remove_path(path: Path, root: Path, dry_run: bool, removed: list[str], skipped: list[str]) -> None:
    if not path.exists():
        skipped.append(str(path))
        return
    if not _is_inside(path, root):
        raise ValueError(f"Refusing to remove path outside package root: {path}")
    removed.append(str(path))
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_many(paths: Iterable[Path], root: Path, dry_run: bool, removed: list[str], skipped: list[str]) -> None:
    seen: set[Path] = set()
    for path in paths:
        p = path.resolve()
        if p in seen:
            continue
        seen.add(p)
        _remove_path(path, root, dry_run, removed, skipped)


def reset_learned_knowledge(
    cfg,
    *,
    dry_run: bool = False,
    include_reports: bool = True,
    include_prepared_knowledge: bool = False,
    recreate_database: bool = True,
) -> dict:
    """Reset learned TDDT knowledge without touching simulators or raw datasets.

    Default scope removes only episode-learned artifacts: runtime SQLite,
    SETD-KStore, SS-KStore training segments, RL state snapshot, and reports. Prepared SARG/CCLL reference
    libraries are preserved because they are deterministic prepared knowledge,
    not learned episode memory. Use include_prepared_knowledge=True to rebuild
    those libraries from scratch on the next prepare step.
    """
    root = Path(cfg.project_root)
    removed: list[str] = []
    skipped: list[str] = []

    sqlite_path = Path(cfg.sqlite_path)
    setd_dir = Path(cfg.setd_kstore_dir)
    rl_state = Path(cfg.rl_state_path)
    report_dir = Path(cfg.report_dir)
    ss_dir = Path(getattr(cfg, "ss_kstore_dir", Path(cfg.project_root) / "working" / "ss_kstore"))

    paths: list[Path] = [sqlite_path, setd_dir, rl_state, ss_dir]
    if include_reports:
        paths.append(report_dir)

    if include_prepared_knowledge:
        paths.extend([
            Path(cfg.ccll_library_json),
            Path(cfg.ccll_daily_descriptors_csv),
            Path(cfg.prepared_ccll_5m_csv),
            Path(cfg.sarg_reference_library_json),
            Path(cfg.sarg_reference_programs_csv),
            Path(cfg.sarg_diet_phase_topk_policy_json),
            Path(cfg.prepared_dir) / "sarg_growth_reference_daily.csv",
            Path(cfg.prepared_dir) / "sarg_growth_reference_input_daily.csv",
            Path(cfg.prepared_dir) / "sarg_growth_simulator_runs",
        ])

    _remove_many(paths, root, dry_run, removed, skipped)

    recreated = None
    if recreate_database and not dry_run:
        store = SQLiteStore(sqlite_path)
        store.close()
        recreated = str(sqlite_path)

    return ResetResult(
        reset_scope="learned_episode_knowledge" + ("+prepared_reference_knowledge" if include_prepared_knowledge else ""),
        dry_run=dry_run,
        removed_paths=removed,
        skipped_paths=skipped,
        recreated_database=recreated,
        preserved_prepared_knowledge=not include_prepared_knowledge,
        status="DRY_RUN" if dry_run else "OK",
    ).to_dict()
