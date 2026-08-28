from pathlib import Path

from ligaps_growth_library import observed_until_day, run_growth_cycle


def test_observed_until_day_sample():
    base = Path(__file__).resolve().parents[1]
    sample = base / "FRACHA19982012_growth_input_1000d_observed_until_day_150.csv"
    assert observed_until_day(sample) == 150


def test_run_growth_cycle_tiny_fixture(tmp_path):
    base = Path(__file__).resolve().parents[1]
    sample = base / "FRACHA19982012_growth_input_1000d_observed_until_day_150.csv"
    tiny = tmp_path / "tiny_growth_input.csv"
    lines = sample.read_text().splitlines()
    tiny.write_text("\n".join(lines[:6]) + "\n")
    result = run_growth_cycle(
        tiny,
        breed=1,
        diet=1,
        scale=1,
        case_id=1,
        output_dir=tmp_path / "out",
        observed_only=True,
        timeout_seconds=60,
    )
    assert len(result.daily) == 5
    assert result.daily["tbw_kg"].notna().any()
    assert "feed_intake_kg_dm_day" in result.daily.columns
    assert result.summary.iloc[0]["observed_until_day"] == 5
