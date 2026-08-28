from pathlib import Path
import pandas as pd

from ligaps_growth_library import run_growth_endOf_cycle

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "FRACHA19982012_growth_input_1000d_observed_until_day_560.csv"

# Example: DataFrame-based end-of-cycle API for RL/growth optimizer loops.
# Configuration used for the Hereford steer individual-mode validation case:
# breed=5, diet=1, scale=1, sex_animal=0, housing=0, case_id=3.
growth_df = pd.read_csv(INPUT_CSV)
last_day = run_growth_endOf_cycle(
    growth_df,
    output_dir=BASE_DIR / "growth_outputs_endof",
    breed=6,
    diet=5,
    scale=1,
    sex_animal=0,
    housing=0,       # 0/"indoor", 1/"outdoor", 2/"open_feedlot", (0,1,0), or "phased"
    case_id=5,
    generate_report=True,
    keep_output_files=True,
)

print("\nEnd-of-cycle last-day dictionary:")
print(last_day)
