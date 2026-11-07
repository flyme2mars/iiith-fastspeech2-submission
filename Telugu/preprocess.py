#!/usr/bin/env python3
"""Assignment entrypoint. Delegates to scripts/preprocess.py with FS2_LANG=telugu."""
import os
import runpy
import sys
from pathlib import Path

os.environ["FS2_LANG"] = "telugu"
script = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
