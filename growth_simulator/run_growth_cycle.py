from pathlib import Path
import pandas as pd

from ligaps_growth_library import run_growth_cycle, run_growth_endOf_cycle

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "FRACHA19982012_growth_input_1000d_observed_until_day_560.csv"

# Example 1: file-based API. It returns the full daily table and can generate the HTML report.
result = run_growth_cycle(
    input_csv=INPUT_CSV,
    output_dir=BASE_DIR / "growth_outputs",
    breed=1,
    diet=1,
    scale=1,
    case_id=1,
    housing="phased",
    generate_report=True,
)

print("Full-cycle output:")
print(result.daily.head())
print(result.summary)
print(f"Offline HTML report: {result.report_html}")

