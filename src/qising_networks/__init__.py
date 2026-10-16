from datetime import date
from pathlib import Path

# Paths from the package location, independent of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"


def run_dir() -> Path:
    """data/<YYYY-MM-DD>/ for a script run today, created if needed. The
    scripts write there; the notebooks read the copy placed in data/."""
    path = DATA_DIR / date.today().isoformat()
    path.mkdir(parents=True, exist_ok=True)
    return path
