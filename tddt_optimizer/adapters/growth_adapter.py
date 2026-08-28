from __future__ import annotations
from pathlib import Path
import sys
import os
import tempfile
import pandas as pd


class GrowthAdapter:
    def __init__(self, project_root: str | Path, tmp_dir: str | Path | None = None):
        self.project_root = Path(project_root)
        self.growth_root = self.project_root / "growth_simulator"
        self.tmp_dir = Path(tmp_dir or "/tmp/tddt_livestock/")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        if str(self.growth_root) not in sys.path:
            sys.path.insert(0, str(self.growth_root))
        from ligaps_growth_library import run_growth_endOf_cycle
        self.run_growth_endOf_cycle = run_growth_endOf_cycle

    def run_end_of_cycle(self, growth_df: pd.DataFrame, output_dir: str | Path, breed: int=6, diet: int=2, scale: int=1, sex_animal: int=0, housing: int=0, case_id: int=1, keep_output_files: bool=True) -> dict:
        # Pandas 3.x is stricter about assigning float transformations into integer columns.
        # Ensure LiGAPS climate columns are serialized as float-valued CSV columns before
        # the unchanged monolithic simulator reads them.
        growth_df = growth_df.copy()
        for col in ["rad", "mint", "maxt", "vpr", "wind", "rain", "aha", "okta"]:
            if col in growth_df.columns:
                growth_df[col] = pd.to_numeric(growth_df[col], errors="coerce").astype(float)
        old_env_tmp = os.environ.get("TMPDIR")
        old_tempdir = tempfile.tempdir
        os.environ["TMPDIR"] = str(self.tmp_dir)
        tempfile.tempdir = str(self.tmp_dir)
        try:
            return self.run_growth_endOf_cycle(
                growth_df,
                output_dir=Path(output_dir),
                breed=breed,
                diet=diet,
                scale=scale,
                sex_animal=sex_animal,
                housing=housing,
                case_id=case_id,
                observed_only=True,
                imax=len(growth_df),
                timeout_seconds=120,
                generate_report=False,
                keep_output_files=keep_output_files,
            )
        finally:
            if old_env_tmp is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = old_env_tmp
            tempfile.tempdir = old_tempdir
