#!/usr/bin/env python3
"""Extract mel, pitch (pyworld), energy, and duration from TextGrids."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ming024 preprocessor (pitch, energy, duration, mel) "
        "using the fixed train/validation manifest."
    )
    parser.add_argument("--lang", choices=["english", "telugu"], default=None)
    return parser.parse_args()


def _phones_in_metadata(preprocessed):
    phones = []
    for name in ("train.txt", "val.txt"):
        path = preprocessed / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _base, _speaker, text, _raw = line.split("|", 3)
            body = text[1:-1] if text.startswith("{") and text.endswith("}") else text
            phones.extend(body.split())
    return phones


def _write_extra_symbols(path, phones):
    existing = []
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    present = set(existing)
    for phone in sorted(set(phones)):
        symbol = "@" + phone
        if symbol not in present:
            existing.append(symbol)
            present.add(symbol)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return existing


def _check_lengths(preprocessed, extra_symbols, cleaners):
    from config_resolve import FS2_ROOT, fs2_env

    code = r"""
import os, sys
from pathlib import Path
import numpy as np
from text import text_to_sequence
pre = Path(os.environ["FS2_PRE"])
cleaners = [item for item in os.environ["FS2_CLEANERS"].split(",") if item]
bad = []
for name in ("train.txt", "val.txt"):
    for line in (pre / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        base, speaker, phones, _raw = line.split("|", 3)
        sequence = text_to_sequence(phones, cleaners)
        duration = np.load(pre / "duration" / "{}-duration-{}.npy".format(speaker, base))
        if len(sequence) != len(duration):
            bad.append((base, len(sequence), int(len(duration))))
            if len(bad) >= 12:
                break
if bad:
    print("Phone/duration length mismatches (basename, n_symbols, n_frames):")
    for item in bad:
        print(" ", item)
    sys.exit(1)
print("Phone and duration lengths match.")
"""
    env = fs2_env(extra_symbols)
    env["FS2_PRE"] = str(preprocessed)
    env["FS2_CLEANERS"] = ",".join(cleaners)
    subprocess.run([sys.executable, "-c", code], cwd=str(FS2_ROOT), env=env, check=True)


def main():
    args = parse_args()
    from config_resolve import FS2_ROOT, REPO_ROOT, load_config, resolve_lang, upstream_configs

    lang = resolve_lang(args.lang)
    config = load_config(lang)
    manifest = REPO_ROOT / config["paths"]["manifest"]
    preprocessed = REPO_ROOT / config["paths"]["preprocessed"]
    if not manifest.is_file():
        raise SystemExit("Missing manifest {}. Download and align first.".format(manifest))
    if not (preprocessed / "TextGrid").is_dir():
        raise SystemExit("Missing TextGrids under {}. Run scripts/align.py first.".format(preprocessed))

    fs2 = str(FS2_ROOT)
    if fs2 not in sys.path:
        sys.path.insert(0, fs2)
    from preprocessor.preprocessor import Preprocessor

    preprocess, _model, _train = upstream_configs(config)
    Preprocessor(preprocess).build_from_path()
    phones = _phones_in_metadata(preprocessed)
    extra_path = preprocessed / "extra_symbols.json"
    _write_extra_symbols(extra_path, phones)
    _check_lengths(preprocessed, extra_path, config["text"]["text_cleaners"])
    print("Features are in {}".format(preprocessed))


if __name__ == "__main__":
    main()
