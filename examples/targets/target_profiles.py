"""Target profile compatibility module.

The canonical implementation lives in `examples/python/tvm_prep/targets.py`.
This file exists so documentation can point to a short, obvious target-profile path.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXAMPLES = ROOT / "python"
if str(PYTHON_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHON_EXAMPLES))

from tvm_prep.targets import TARGET_PROFILES, TargetProfile, get_target_profile, list_target_profiles

__all__ = ["TARGET_PROFILES", "TargetProfile", "get_target_profile", "list_target_profiles"]
