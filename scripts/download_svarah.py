#!/usr/bin/env python3
"""Download a ~5 h / ~1 h English subset from the gated Svarah dataset.

This does not run inside the cloud-agent scaffold. Run it locally or on Kaggle
after accepting the dataset and setting HF_TOKEN.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


SPEAKER_FIELDS = [
    "gender",
    "age-group",
    "primary_language",
    "native_place_state",
    "native_place_district",
    "highest_qualification",
    "job_category",
    "occupation_domain",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and export a Svarah subset for English FastSpeech 2. "
        "The dataset is gated: accept it on the Hugging Face page and set HF_TOKEN."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if the manifest exists.")
    parser.add_argument("--limit", type=int, default=0, help="Debug: keep at most this many clips before the hour budget.")
    return parser.parse_args()


def speaker_key(row):
    """Proxy id. The HF release has demographics but no speaker_id column."""
    if not any(field in row for field in SPEAKER_FIELDS):
        return "svarah_pooled"
    raw = "|".join(str(row.get(field, "")).strip().lower() for field in SPEAKER_FIELDS)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    parts = []
    for field in ("gender", "primary_language", "native_place_state"):
        piece = re.sub(r"[^a-z0-9]+", "", str(row.get(field, "")).strip().lower())
        if piece:
            parts.append(piece[:12])
    prefix = "_".join(parts) or "spk"
    return "{}_{}".format(prefix, digest)[:80]


def main():
    args = parse_args()
    from config_resolve import REPO_ROOT, load_config
    from hf_audio import require_token
    from materialize import write_subset
    from subset import select_subset

    config = load_config("english")
    manifest = REPO_ROOT / config["paths"]["manifest"]
    if manifest.is_file() and not args.force:
        print("Manifest already exists at {}. Pass --force to rebuild.".format(manifest))
        return

    token = require_token(config["hf_dataset"])
    from datasets import Audio, load_dataset

    print("Loading {} ({}) with the HF token. Audio stays encoded until a row is selected.".format(
        config["hf_dataset"], config["hf_split"]
    ))
    dataset = load_dataset(config["hf_dataset"], split=config["hf_split"], token=token)
    audio_column = "audio_filepath" if "audio_filepath" in dataset.column_names else "audio"
    dataset = dataset.cast_column(audio_column, Audio(decode=False))

    missing = [name for name in ("text", "duration") if name not in dataset.column_names]
    if missing or audio_column not in dataset.column_names:
        raise SystemExit(
            "Unexpected Svarah columns {}. Need text, duration, and {}.".format(
                dataset.column_names, audio_column
            )
        )

    subset = config["subset"]
    texts = dataset["text"]
    durations = dataset["duration"]
    columns = {name: dataset[name] for name in SPEAKER_FIELDS if name in dataset.column_names}
    if not columns:
        print(
            "WARNING: Svarah speaker demographics were not found. "
            "Every clip will share the speaker id svarah_pooled."
        )

    candidates = []
    for index, (text, duration) in enumerate(zip(texts, durations)):
        text = "" if text is None else str(text).strip()
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            continue
        if not text or not (subset["min_utt_sec"] <= duration <= subset["max_utt_sec"]):
            continue
        row = {name: values[index] for name, values in columns.items()}
        candidates.append(
            {
                "id": "en_{:05d}".format(index),
                "speaker": speaker_key(row),
                "duration": duration,
                "text": text,
                "source_index": index,
                "meta": {name: "" if row.get(name) is None else str(row.get(name)) for name in row},
            }
        )
        if args.limit and len(candidates) >= args.limit:
            break

    print("Candidate clips after duration/text filters: {}".format(len(candidates)))
    train, val, stats = select_subset(
        candidates,
        train_seconds=float(subset["train_hours"]) * 3600.0,
        val_seconds=float(subset["val_hours"]) * 3600.0,
        min_speaker_seconds=float(subset["min_speaker_sec"]),
        seed=int(subset["seed"]),
    )
    chosen = {row["id"]: row for row in train}
    for row in val:
        chosen[row["id"]] = row
    split_of = {row["id"]: "train" for row in train}
    split_of.update({row["id"]: "val" for row in val})

    def lab_text(text):
        fs2 = str(REPO_ROOT / "third_party" / "FastSpeech2")
        if fs2 not in sys.path:
            sys.path.insert(0, fs2)
        from text.cleaners import english_cleaners

        return english_cleaners(text)

    rows = []
    for item in list(train) + list(val):
        audio = dataset[item["source_index"]][audio_column]
        rows.append(
            {
                "id": item["id"],
                "speaker": item["speaker"],
                "split": split_of[item["id"]],
                "text": item["text"],
                "audio": audio,
                "meta": item["meta"],
            }
        )

    stats.update(
        {
            "dataset": config["hf_dataset"],
            "split_name_on_hub": config["hf_split"],
            "note": (
                "Svarah publishes a single 'test' split. The train/val split in this "
                "manifest is created here. speaker ids are a hash of demographic fields "
                "because the hub release has no speaker_id. Distinct people who share "
                "those fields are merged."
            ),
            "selection": {
                "train_hours": subset["train_hours"],
                "val_hours": subset["val_hours"],
                "min_utt_sec": subset["min_utt_sec"],
                "max_utt_sec": subset["max_utt_sec"],
                "min_speaker_sec": subset["min_speaker_sec"],
                "seed": subset["seed"],
                "candidates": len(candidates),
            },
            "reached_target": stats["reached_target"],
        }
    )
    write_subset(
        rows,
        REPO_ROOT / config["paths"]["raw"],
        manifest,
        manifest.parent / "split_stats.json",
        int(config["audio"]["sampling_rate"]),
        lab_text,
        stats,
    )


if __name__ == "__main__":
    main()
