#!/usr/bin/env python3
"""Plot predicted duration, F0, and energy for each test sentence.

Reads the ``.npz`` files written by scripts/synthesize.py. It does not
synthesize on its own and does not draw empty axes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot variance-adaptor predictions from synthesize.py.")
    parser.add_argument("--lang", choices=["english", "telugu"], default=None)
    return parser.parse_args()


def plot_sentence(npz_path, output_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    payload = np.load(npz_path, allow_pickle=True)
    phones = [str(item) for item in payload["phones"].tolist()]
    duration = np.asarray(payload["duration"], dtype=float)
    pitch = np.asarray(payload["pitch"], dtype=float)
    energy = np.asarray(payload["energy"], dtype=float)
    sentence_id = str(payload["sentence_id"])
    indexes = list(range(len(phones)))
    panels = {
        "duration": (duration, "predicted duration (mel frames)"),
        "pitch": (pitch, "predicted F0 (Hz, denormalized)"),
        "energy": (energy, "predicted energy (denormalized)"),
    }
    written = []
    for folder, (values, ylabel) in panels.items():
        if len(values) != len(phones):
            raise SystemExit(
                "{} has {} values for {} phones.".format(npz_path, len(values), len(phones))
            )
        figure, axis = plt.subplots(figsize=(max(6, len(phones) * 0.28), 3.6))
        if folder == "duration":
            axis.bar(indexes, values, color="#3d5a40")
        else:
            axis.plot(indexes, values, color="#9b2226" if folder == "pitch" else "#1d3557")
        axis.set_xticks(indexes)
        axis.set_xticklabels(phones, rotation=90, fontsize=7)
        axis.set_ylabel(ylabel)
        axis.set_title("{} {}".format(sentence_id, folder))
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        destination = output_root / folder / "{}.png".format(sentence_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=120)
        plt.close(figure)
        written.append(destination)
    return written


def main():
    args = parse_args()
    from config_resolve import REPO_ROOT, load_config, resolve_lang

    lang = resolve_lang(args.lang)
    config = load_config(lang)
    prediction_dir = REPO_ROOT / config["paths"]["result"] / "variance_adaptor_plots" / "_predictions"
    files = sorted(prediction_dir.glob("*.npz")) if prediction_dir.is_dir() else []
    if not files:
        raise SystemExit(
            "No prediction files in {}. Run scripts/synthesize.py --lang {} first.".format(
                prediction_dir, lang
            )
        )
    output_root = REPO_ROOT / config["paths"]["result"] / "variance_adaptor_plots"
    count = 0
    for path in files:
        written = plot_sentence(path, output_root)
        count += 1
        for item in written:
            print("Wrote {}".format(item))
    print("Plotted {} sentences.".format(count))


if __name__ == "__main__":
    main()
