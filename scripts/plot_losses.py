#!/usr/bin/env python3
"""Write training_loss.png and validation_loss.png from the trainer logs.

Exits with an error if a log is missing. It does not draw a placeholder curve.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TRAIN_RE = re.compile(
    r"Step (\d+)/\d+, Total Loss: ([0-9.eE+-]+), Mel Loss: ([0-9.eE+-]+), "
    r"Mel PostNet Loss: ([0-9.eE+-]+), Pitch Loss: ([0-9.eE+-]+), "
    r"Energy Loss: ([0-9.eE+-]+), Duration Loss: ([0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"Validation Step (\d+), Total Loss: ([0-9.eE+-]+), Mel Loss: ([0-9.eE+-]+), "
    r"Mel PostNet Loss: ([0-9.eE+-]+), Pitch Loss: ([0-9.eE+-]+), "
    r"Energy Loss: ([0-9.eE+-]+), Duration Loss: ([0-9.eE+-]+)"
)
COLUMNS = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot FastSpeech 2 train and validation losses.")
    parser.add_argument("--lang", choices=["english", "telugu"], default=None)
    return parser.parse_args()


def _from_csv(path):
    steps = []
    series = {name: [] for name in COLUMNS}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            for name in COLUMNS:
                series[name].append(float(row[name]))
    return steps, series


def _from_log(path, pattern):
    steps = []
    series = {name: [] for name in COLUMNS}
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in pattern.finditer(text):
        steps.append(int(match.group(1)))
        for index, name in enumerate(COLUMNS, start=2):
            series[name].append(float(match.group(index)))
    return steps, series


def _load_train(log_dir):
    csv_path = log_dir / "losses.csv"
    if csv_path.is_file():
        return _from_csv(csv_path)
    log_path = log_dir / "log.txt"
    if log_path.is_file():
        return _from_log(log_path, TRAIN_RE)
    return [], {}


def _load_val(log_dir):
    log_path = log_dir / "log.txt"
    if not log_path.is_file():
        return [], {}
    return _from_log(log_path, VAL_RE)


def _draw(steps, series, title, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name in COLUMNS:
        axis.plot(steps, series[name], label=name)
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title(title)
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main():
    args = parse_args()
    from config_resolve import REPO_ROOT, load_config, resolve_lang

    lang = resolve_lang(args.lang)
    config = load_config(lang)
    log_root = REPO_ROOT / config["paths"]["log"]
    results = REPO_ROOT / config["paths"]["result"]
    train_steps, train_series = _load_train(log_root / "train")
    val_steps, val_series = _load_val(log_root / "val")
    missing = []
    if len(train_steps) < 1:
        missing.append(str(log_root / "train"))
    else:
        destination = results / "training_loss.png"
        _draw(train_steps, train_series, "{} training loss".format(lang), destination)
        print("Wrote {}".format(destination))
    if len(val_steps) < 1:
        missing.append(str(log_root / "val"))
    else:
        destination = results / "validation_loss.png"
        _draw(val_steps, val_series, "{} validation loss".format(lang), destination)
        print("Wrote {}".format(destination))
    if missing:
        raise SystemExit(
            "No loss rows in {}. Train longer (val_step is {}) before plotting. "
            "This script does not invent curves.".format(
                ", ".join(missing), config["step"]["val_step"]
            )
        )


if __name__ == "__main__":
    main()
