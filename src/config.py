# config.py
import warnings
from curl_cffi import requests

# -----------------------------
# General Settings
# -----------------------------
warnings.filterwarnings("ignore")
session = requests.Session(impersonate="chrome")

# -----------------------------
# Portfolio Settings
# -----------------------------
TICKERS = ["SPY", "QQQ", "VTI", "EFA", "TLT", "SHY", "TIP"]
RF_RATE = 0.04  # risk-free rate
NUM_PORTFOLIOS = 1_000  # Monte Carlo portfolios per subset
WINDOW_SIZE = 252 * 10  # 10-year rolling window
STEP_SIZE = 252  # 1-year step
DATA_PATH = "data/data.csv"

# -----------------------------
# Portfolio Settings
# -----------------------------
MED_START_DATE = "2005-10-24"
MED_END_DATE = "2025-10-24"
