"""Library interface for using the LiGAPS-Beef Python port inside a growth optimizer/RL loop.

The original translated model is a monolithic script.  This module keeps that script
intact and exposes a stable function that runs one daily growth-optimizer cycle:

    run_growth_cycle(input_csv=..., breed=1, diet=1, observed_only=True)

The input CSV can use the growth-optimizer schema:
    fattening_day, yr, doy, rad, mint, maxt, vpr, wind, rain, aha, okta,
    is_observed, source_day

Only rows up to the last row with is_observed == 1 are simulated by default.  The
function returns pandas DataFrames containing the daily trajectory and the summary.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from ligaps_growth_report import generate_growth_html_report


PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_SCRIPT = PACKAGE_DIR / "LiGAPSBeef20180301_herd_worked.py"


@dataclass(frozen=True)
class GrowthCycleResult:
    """Result returned by :func:`run_growth_cycle`.

    Attributes
    ----------
    daily:
        Daily simulated trajectory up to the observed horizon.
    summary:
        One-row summary table with metadata and aggregate outputs.
    output_dir:
        Directory containing the generated CSV/JSON files and any model plots.
    returncode:
        Subprocess return code from the underlying LiGAPS-Beef script.
    stdout:
        Captured standard output from the underlying script.
    stderr:
        Captured standard error from the underlying script.
    report_html:
        Path to the generated offline HTML report.
    """

    daily: pd.DataFrame
    summary: pd.DataFrame
    output_dir: Path
    returncode: int
    stdout: str
    stderr: str
    report_html: Optional[Path] = None


def observed_until_day(input_csv: str | Path) -> int:
    """Return the last observed fattening day in a growth-optimizer CSV."""
    df = pd.read_csv(input_csv)
    lower = {str(c).strip().lower(): c for c in df.columns}
    if "is_observed" not in lower:
        return int(len(df))
    obs = pd.to_numeric(df[lower["is_observed"]], errors="coerce").fillna(0)
    idx = obs[obs == 1].index
    return int(idx[-1] + 1) if len(idx) else 0


def _housing_code(value: int | str) -> int:
    """Return LiGAPS housing code: 0=indoor/stable, 1=outdoor, 2=open feedlot."""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"0", "indoor", "inside", "stable", "housed", "housing", "feedlot", "closed"}:
        return 0
    if text in {"1", "outdoor", "outside", "free_grazing", "grazing", "pasture", "open_air"}:
        return 1
    if text in {"2", "open_feedlot", "openfeedlot"}:
        return 2
    raise ValueError(f"Unsupported housing code: {value!r}; use 0/'indoor', 1/'outdoor', or 2/'open_feedlot'.")


def _housing_env(
    housing: int | str | Sequence[int] | None = None,
    housing_phase1: int | None = None,
    housing_phase2: int | None = None,
    housing_phase3: int | None = None,
) -> dict[str, str]:
    """Convert API housing arguments to model environment variables.

    Housing codes follow the reference model: 0 = stable/feedlot,
    1 = outdoor/free grazing, 2 = open feedlot. ``housing`` may be a
    single code/string for all days or a three-item phase sequence.
    Accepted strings include ``indoor``, ``outdoor``, ``open_feedlot``,
    ``phased`` and ``mixed``.
    """
    env: dict[str, str] = {}
    phase_values = [housing_phase1, housing_phase2, housing_phase3]
    if housing is not None:
        if isinstance(housing, (list, tuple)):
            if len(housing) != 3:
                raise ValueError("housing sequence must contain exactly three phase codes")
            env["LIGAPS_HOUSING_PHASES"] = ",".join(str(_housing_code(x)) for x in housing)
        elif isinstance(housing, int):
            env["LIGAPS_HOUSING_MODE"] = str(int(housing))
        else:
            env["LIGAPS_HOUSING_MODE"] = str(housing)
    if any(v is not None for v in phase_values):
        if not all(v is not None for v in phase_values):
            raise ValueError("housing_phase1, housing_phase2 and housing_phase3 must be provided together")
        env["LIGAPS_HOUSING_PHASES"] = ",".join(str(_housing_code(v)) for v in phase_values if v is not None)
    return env


def _sex_code(value: int | str) -> int:
    """Return LiGAPS sex code: 0=male, 1=female."""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"0", "male", "m", "bull", "steer"}:
        return 0
    if text in {"1", "female", "f", "cow", "heifer"}:
        return 1
    raise ValueError(f"Unsupported sex_animal value: {value!r}; use 0/'male' or 1/'female'.")


def run_growth_cycle(
    input_csv: str | Path,
    *,
    breed: int = 1,
    diet: int = 1,
    scale: int = 1,
    sex_animal: int | str = 0,
    case_id: int = 1,
    housing: int | str | Sequence[int] | None = None,
    housing_phase1: int | None = None,
    housing_phase2: int | None = None,
    housing_phase3: int | None = None,
    output_dir: str | Path | None = None,
    observed_only: bool = True,
    imax: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    extra_env: Optional[dict[str, str]] = None,
    generate_report: bool = True,
) -> GrowthCycleResult:
    """Run LiGAPS-Beef for one RL/growth-optimizer cycle.

    Parameters
    ----------
    input_csv:
        Daily climate/hall summary file.  The function accepts the columns
        ``fattening_day, yr, doy, rad, mint, maxt, vpr, wind, rain, aha, okta,
        is_observed, source_day, season_start_date``. Extra columns are ignored,
        while date metadata is propagated to the daily output when present.
    breed:
        LiGAPS breed code.  For the original model examples: ``1`` is Charolais,
        ``4`` is 3/4 Brahman x 1/4 Shorthorn, ``5`` is Hereford, and ``6`` is Holstein/Holstein-Friesian.
    diet:
        LiGAPS diet number (1..5), matching ``FEEDNR`` in the reference code.
    scale:
        ``1`` for individual-animal mode, ``2`` for herd-unit mode.  RL use is
        normally ``1``.
    sex_animal:
        Animal sex used in individual-animal mode: ``0``/``male`` = male, ``1``/``female`` = female.
    case_id:
        Case slot to reuse in the monolithic script. Use ``1`` unless you need
        one of the original illustration cases.
    housing:
        Housing schedule. Use ``0``/``indoor`` for stable/feedlot all days,
        ``1``/``outdoor`` for free-grazing/outdoor all days, ``2``/``open_feedlot``
        for open feedlot all days, ``phased``/``mixed`` for configured phases,
        or a three-item sequence such as ``(0, 1, 0)`` for phase1/phase2/phase3.
    housing_phase1, housing_phase2, housing_phase3:
        Optional explicit phase codes. If one is provided, all three must be provided.
    output_dir:
        Directory where daily/summary outputs are written.  If omitted, a
        temporary directory is created and retained for inspection.
    observed_only:
        If true, simulate only through the last row where ``is_observed == 1``.
    imax:
        Optional hard maximum horizon.  If omitted, the model uses its default,
        which is then clipped to the observed horizon when ``observed_only=True``.
    timeout_seconds:
        Optional timeout for the subprocess execution.
    extra_env:
        Optional additional environment variables for advanced testing.
    generate_report:
        If true, create the offline HTML report in ``output_dir``. Set this to
        false for RL/training calls that only need numeric outputs.

    Returns
    -------
    GrowthCycleResult
        Daily trajectory, summary, output paths, and captured process logs.
    """
    input_csv = Path(input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ligaps_growth_cycle_"))
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.update(
        {
            "LIGAPS_GROWTH_INPUT_CSV": str(input_csv),
            "LIGAPS_OUTPUT_DIR": str(output_dir),
            "LIGAPS_CASE_IDS": str(int(case_id)),
            "LIGAPS_BREED": str(int(breed)),
            "LIGAPS_DIET": str(int(diet)),
            "LIGAPS_SCALE": str(int(scale)),
            "LIGAPS_SEX_ANIMAL": str(_sex_code(sex_animal)),
            "LIGAPS_OBSERVED_ONLY": "1" if observed_only else "0",
            "LIGAPS_FORCE_EXIT": "1",
        }
    )
    env.update(_housing_env(housing, housing_phase1, housing_phase2, housing_phase3))
    if imax is not None:
        env["LIGAPS_IMAX"] = str(int(imax))
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    
    # Run the monolithic model with -S and an explicit PYTHONPATH. This avoids
    # expensive user-site startup hooks and keeps repeated RL calls deterministic.
    _site_paths = [p for p in sys.path if p and "site-packages" in p]
    if _site_paths:
        env["PYTHONPATH"] = os.pathsep.join(_site_paths + ([env.get("PYTHONPATH")] if env.get("PYTHONPATH") else []))
    proc = subprocess.run(
        [sys.executable, "-S", str(MODEL_SCRIPT)],
        cwd=str(PACKAGE_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "LiGAPS-Beef growth cycle failed with return code "
            f"{proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    daily_path = output_dir / f"growth_optimizer_outputs_case_{int(case_id)}.csv"
    summary_path = output_dir / f"growth_optimizer_summary_case_{int(case_id)}.json"
    if not daily_path.exists():
        raise FileNotFoundError(f"Expected daily output was not created: {daily_path}")
    daily = pd.read_csv(daily_path)
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_obj = json.load(f)
        summary = pd.DataFrame([summary_obj])
    else:
        summary = pd.DataFrame()

    report_html = None
    if generate_report:
        report_html = generate_growth_html_report(
            daily,
            summary,
            output_dir,
            case_id=int(case_id),
        )

    return GrowthCycleResult(
        daily=daily,
        summary=summary,
        output_dir=output_dir,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        report_html=report_html,
    )


def _coerce_last_day_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def run_growth_endOf_cycle(
    growth_df: pd.DataFrame,
    *,
    breed: int = 1,
    diet: int = 1,
    scale: int = 1,
    sex_animal: int | str = 0,
    case_id: int = 1,
    housing: int | str | Sequence[int] | None = None,
    housing_phase1: int | None = None,
    housing_phase2: int | None = None,
    housing_phase3: int | None = None,
    output_dir: str | Path | None = None,
    observed_only: bool = True,
    imax: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    extra_env: Optional[dict[str, str]] = None,
    generate_report: bool = False,
    keep_output_files: bool = False,
) -> dict[str, Any]:
    """Run one LiGAPS-Beef cycle from an in-memory DataFrame and return only the last day.

    This API is intended for reinforcement-learning / growth-optimizer loops.
    The optimizer passes the climate dataset directly as a ``pandas.DataFrame``
    instead of writing a CSV itself. Internally, the unchanged monolithic
    simulator still receives a temporary CSV file.

    Parameters
    ----------
    growth_df:
        Input climate/hall DataFrame with the growth-optimizer columns
        ``fattening_day, yr, doy, rad, mint, maxt, vpr, wind, rain, aha, okta,
        is_observed, source_day, season_start_date``. Extra columns are preserved
        in the temporary CSV; date metadata is propagated into the daily output.
    generate_report:
        If true, create the offline HTML report in ``output_dir``. If false, no
        HTML report is generated. This should normally remain false during RL
        training loops.
    keep_output_files:
        If true, keep the generated daily CSV/summary JSON and optional report
        in ``output_dir``. If false, temporary files are deleted after the
        last-day dictionary has been created.

    Returns
    -------
    dict
        The final row of ``growth_optimizer_outputs_case_<case_id>.csv`` as a
        plain Python dictionary. No full daily trajectory is returned.
    """
    if not isinstance(growth_df, pd.DataFrame):
        raise TypeError("growth_df must be a pandas.DataFrame")
    if growth_df.empty:
        raise ValueError("growth_df is empty")

    temp_root = Path(tempfile.mkdtemp(prefix="ligaps_endof_cycle_"))
    temp_input = temp_root / "growth_input_dataframe.csv"
    growth_df.to_csv(temp_input, index=False)

    if keep_output_files:
        output_path = Path(output_dir).resolve() if output_dir is not None else PACKAGE_DIR / "growth_outputs"
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = temp_root / "outputs"

    try:
        result = run_growth_cycle(
            input_csv=temp_input,
            breed=breed,
            diet=diet,
            scale=scale,
            sex_animal=sex_animal,
            case_id=case_id,
            housing=housing,
            housing_phase1=housing_phase1,
            housing_phase2=housing_phase2,
            housing_phase3=housing_phase3,
            output_dir=output_path,
            observed_only=observed_only,
            imax=imax,
            timeout_seconds=timeout_seconds,
            extra_env=extra_env,
            generate_report=generate_report,
        )
        if result.daily.empty:
            raise RuntimeError("LiGAPS-Beef produced an empty daily output table")
        last = result.daily.iloc[-1]
        last_day = {str(k): _coerce_last_day_value(v) for k, v in last.to_dict().items()}
    finally:
        if not keep_output_files:
            shutil.rmtree(temp_root, ignore_errors=True)
        else:
            # Remove only the temporary input folder. The requested output folder is preserved.
            shutil.rmtree(temp_root, ignore_errors=True)

    return last_day


__all__ = ["GrowthCycleResult", "observed_until_day", "run_growth_cycle", "run_growth_endOf_cycle", "generate_growth_html_report"]
