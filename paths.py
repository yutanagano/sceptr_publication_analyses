from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = (PROJECT_ROOT/"analysis_results").resolve(strict=True)
DATA_DIR = (PROJECT_ROOT/"tcr_data").resolve(strict=True)
CACHE_DIR = PROJECT_ROOT/".representation_cache"


CACHE_DIR.mkdir(exist_ok=True)