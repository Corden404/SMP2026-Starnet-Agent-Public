"""Runtime bridge for the action-surrogate experiment.

The host Python has scikit-learn; the project environment has the competition
framework.  Import the compiled ML stack first, then append the framework-only
paths so their separate numpy builds cannot shadow one another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib  # noqa: F401
import numpy  # noqa: F401
import sklearn  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "vendor/casevo/src", ROOT / ".venv/Lib/site-packages"):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.append(str(extra))

from action_surrogate_experiment import main


if __name__ == "__main__":
    main()
