from pathlib import Path

import pandas as pd

from ligaps_growth_library import run_growth_endOf_cycle


def test_run_growth_endof_cycle_accepts_dataframe_and_returns_dict(tmp_path):
    base = Path(__file__).resolve().parents[1]
    src = base / "FRACHA19982012_growth_input_1000d_observed_until_day_150.csv"
    df = pd.read_csv(src).head(5).copy()
    df["is_observed"] = 1

    # This test uses a tiny horizon to validate the API contract without exercising
    # the full biological scenario.
    out = run_growth_endOf_cycle(
        df,
        output_dir=tmp_path / "outputs",
        breed=1,
        diet=1,
        scale=1,
        case_id=1,
        sex_animal="male",
        housing=("indoor", "outdoor", "indoor"),
        generate_report=False,
        keep_output_files=False,
        imax=5,
        timeout_seconds=120,
    )

    assert isinstance(out, dict)
    assert out
    assert "fattening_day" in out
    assert out["sex_animal"] == 0
    assert "growth_start_date" in out
    assert "growth_date" in out


def test_female_endof_cycle_output_has_no_yr_and_no_post_slaughter_collapse(tmp_path):
    base = Path(__file__).resolve().parents[1]
    src = base / "FRACHA19982012_growth_input_1000d_observed_until_day_560.csv"
    df = pd.read_csv(src).head(390).copy()
    df["is_observed"] = 1

    out = run_growth_endOf_cycle(
        df,
        output_dir=tmp_path / "female_outputs",
        breed=4,
        diet=5,
        scale=1,
        case_id=1,
        sex_animal="female",
        housing="indoor",
        generate_report=False,
        keep_output_files=True,
        imax=390,
        timeout_seconds=120,
    )

    csv_path = tmp_path / "female_outputs" / "growth_optimizer_outputs_case_1.csv"
    daily = pd.read_csv(csv_path)
    assert "yr" not in daily.columns
    assert "yr" not in out
    window = daily.loc[daily["fattening_day"].between(379, 390)]
    assert (window["tbw_kg"] > 0).all()
    assert window["beef_production_kg"].notna().all()
    assert (window["me_uptake_mj_day"] > 0).all()
    assert (window["heat_production"] > 0).all()
