from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SYN = ROOT / "data" / "synthetic"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"
SQL = ROOT / "sql"

SEED = 42

# Economic ASSUMPTIONS (not from the data). Currency units are whatever the source uses.
# Margin: net revenue on a repaid loan as a share of AMT_CREDIT. LGD: loss share of AMT_CREDIT on default.
MARGIN = 0.15
LGD = 0.60

# Simulated legacy approval policy used for reject inference: reject the riskiest REJECT_SHARE
# of the "through the door" population on a crude rule (see src/reject_inference.py).
REJECT_SHARE = 0.30

# A/B experiment assumptions
ALPHA = 0.05
POWER = 0.80
WEEKLY_APPLICATIONS = 20000  # assumed traffic for one lender; change to your own
