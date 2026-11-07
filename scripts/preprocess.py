#!/usr/bin/env python3
"""Align (unless skipped) and extract FastSpeech 2 features for one language.

Downloading the dataset is a separate step because Svarah is gated and both
corpora are multi-gigabyte. Pass --download to run that step first.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Align and extract features for one language.")
    parser.add_argument("--lang", choices=["english", "telugu"], default=None)
    parser.add_argument("--download", action="store_true", help="Run the language download script first.")
    parser.add_argument("--aligner", choices=["ctc", "mfa"], default=None)
    parser.add_argument("--skip-align", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _run(script, extra):
    command = [sys.executable, str(Path(__file__).resolve().parent / script)] + extra
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    from config_resolve import resolve_lang

    lang = resolve_lang(args.lang)
    if args.download:
        script = "download_svarah.py" if lang == "english" else "download_indictts.py"
        _run(script, [])
    if not args.skip_align:
        extra = ["--lang", lang]
        if args.aligner:
            extra.extend(["--aligner", args.aligner])
        if args.limit:
            extra.extend(["--limit", str(args.limit)])
        _run("align.py", extra)
    if not args.skip_extract:
        _run("extract_features.py", ["--lang", lang])


if __name__ == "__main__":
    main()
