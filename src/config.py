"""Project-wide constants."""

from pathlib import Path

# ----------------------------------------------------------------------- Paths
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
CACHE_DIR  = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "output"
FIG_DIR    = OUTPUT_DIR / "figures"
APP_DIR    = FIG_DIR / "appendix"
TBL_DIR    = OUTPUT_DIR / "tables"

for d in (CACHE_DIR, FIG_DIR, APP_DIR, TBL_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "data.xlsx"

# ------------------------------------------------------------------- Portfolio
ASSETS              = ["SPY", "AGG", "PE", "NPI"]
WEIGHTS             = [0.40, 0.20, 0.25, 0.15]
EXPECTED_RETURNS    = [0.060, 0.045, 0.095, 0.070]   # annualized
HORIZON_QUARTERS    = 12          # 3-year horizon
ALPHA               = 0.95
QUARTERS_PER_YEAR   = 4

# ----------------------------------------------------------------------- Data
LEVEL_COLUMNS = ["SPY", "AGG", "PE", "NPI", "LPX50", "RMZ", "SPX", "DXY", "BCOM"]
DIFF_COLUMNS  = ["UST10", "UST2",
                 "VIX_eom", "VIX_mean", "VIX_logmean",
                 "HY_OAS_eom", "HY_OAS_mean", "IG_OAS_eom", "IG_OAS_mean"]
PE_PROXY  = "LPX50"
NPI_PROXY = "RMZ"

# ----------------------------------------------------------------- Random seed
# Master seed; sub-modules use deterministic offsets so independent stages can
# be re-run without disturbing each other.
SEED = 42
SEED_SIM_HS        = SEED      # historical simulation
SEED_SIM_HS_UNSM   = SEED + 1
SEED_SIM_FHS       = SEED + 2
SEED_SIM_CHAMPION  = SEED + 3
SEED_COPULA_SIM    = SEED + 100
SEED_BOOT_TIER_A   = SEED + 1000
SEED_BOOT_TIER_B   = SEED + 2000

# --------------------------------------------------------------- MC parameters
N_PATHS_MC          = 1_000_000
BLOCK_BOOTSTRAP_LEN = 4         # mean block length, ≈ 1 year
N_BOOT_TIER_A       = 2_000     # MC-noise bootstrap
N_BOOT_TIER_B       = 500       # parameter bootstrap
N_PATHS_BOOT_TIER_B = 100_000   # MC paths per Tier-B bootstrap iteration

# ------------------------------------------------------------------- Marginals
EWMA_LAMBDA_BOUNDS  = (0.80, 0.99)
EWMA_LAMBDA_FALLBACK = 0.94
SEMIPARAM_LOWER_Q   = 0.10
SEMIPARAM_UPPER_Q   = 0.90

# ---------------------------------------------------------------------- Copula
NU_GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 10.0, 15.0, 30.0]
COPULA_OOS_SPLIT = 60   # in-sample n=60, out-of-sample n=20

# ------------------------------------------------------------ Crisis anchors
NAMED_CRISES = [
    ("2007-09-30", "2010-06-30", "GFC"),
    ("2019-12-31", "2022-09-30", "COVID"),
]
STRESS_SCENARIOS = [
    ("GFC", 1.2, "GFC x 1.2 (Severe Equity Crash)"),
    ("GFC", 1.5, "GFC x 1.5 (Stress)"),
    ("GFC", 2.0, "GFC x 2.0 (Catastrophic)"),
]
