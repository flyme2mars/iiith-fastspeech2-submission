#!/usr/bin/env python3
"""Download a ~5 h / ~1 h Telugu subset from SPRINGLab/IndicTTS_Telugu.

The hub split is already named train and contains both speakers (~8.74 h).
This script builds a fresh train/validation split from it. The dataset is
public; HF_TOKEN is optional.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and export an IndicTTS Telugu subset for FastSpeech 2. "
        "HF_TOKEN is optional. The corpus is public, but it is licensed by IIT Madras "
        "IndicTTS: https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the wavs and the manifest. Reuses duration_index.jsonl unless --reindex is set.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Scan the hub split again and rewrite duration_index.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug: scan at most this many rows.")
    return parser.parse_args()


def speaker_name(gender):
    if isinstance(gender, str):
        lowered = gender.strip().lower()
        if lowered in {"female", "male"}:
            return lowered
        return re_safe(lowered)
    try:
        return ["female", "male"][int(gender)]
    except (TypeError, ValueError, IndexError):
        return "spk"


def re_safe(text):
    import re

    cleaned = re.sub(r"[^a-z0-9]+", "", text)
    return cleaned or "spk"


def main():
    args = parse_args()
    from chunk_text import chunk_utterance
    from config_resolve import REPO_ROOT, load_config
    from hf_audio import audio_header, read_hf_token
    from materialize import write_subset
    from subset import select_subset

    config = load_config("telugu")
    manifest = REPO_ROOT / config["paths"]["manifest"]
    index_path = manifest.parent / "duration_index.jsonl"
    if manifest.is_file() and not args.force:
        print(
            "Manifest already exists at {}. Pass --force to rebuild the subset "
            "(the duration index is reused unless you also pass --reindex).".format(manifest)
        )
        return

    token = read_hf_token()
    from datasets import Audio, load_dataset

    print(
        "Loading {}. A token is not required. IndicTTS license: "
        "https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf".format(config["hf_dataset"])
    )
    dataset = load_dataset(
        config["hf_dataset"],
        split=config["hf_split"],
        token=token,
    )
    if "audio" not in dataset.column_names or "text" not in dataset.column_names:
        raise SystemExit("Unexpected columns: {}".format(dataset.column_names))
    dataset = dataset.cast_column("audio", Audio(decode=False))

    subset = config["subset"]
    if index_path.is_file() and not args.reindex:
        print("Reusing duration index {}".format(index_path))
        catalog = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        catalog = []
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as handle:
            for index, row in enumerate(dataset):
                if args.limit and index >= args.limit:
                    break
                text = "" if row.get("text") is None else str(row["text"]).strip()
                try:
                    duration, sample_rate = audio_header(row["audio"])
                except Exception:
                    continue
                entry = {
                    "source_index": index,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "text": text,
                    "gender": speaker_name(row.get("gender")),
                }
                catalog.append(entry)
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print("Indexed {} Telugu clips.".format(len(catalog)))

    candidates = []
    indexed_seconds = 0.0
    clips_without_chunk = 0
    for entry in catalog:
        text = entry["text"].strip()
        duration = float(entry["duration"])
        indexed_seconds += duration
        chunks = chunk_utterance(
            text,
            duration,
            max_sec=float(subset["max_utt_sec"]),
            min_sec=float(subset["min_utt_sec"]),
        )
        if not chunks:
            clips_without_chunk += 1
            continue
        for chunk_index, chunk in enumerate(chunks):
            candidates.append(
                {
                    "id": "te_{:05d}_{:02d}".format(entry["source_index"], chunk_index),
                    "speaker": entry["gender"],
                    "duration": float(chunk["duration"]),
                    "text": chunk["text"],
                    "source_index": entry["source_index"],
                    "start_sec": float(chunk["start"]),
                    "end_sec": float(chunk["end"]),
                    "meta": {
                        "gender": entry["gender"],
                        "source_sampling_rate": entry.get("sample_rate"),
                        "source_duration_sec": duration,
                    },
                }
            )

    print(
        "Candidate slices after chunking and CTC length filter: {} (from {} indexed clips, {} dropped).".format(
            len(candidates), len(catalog), clips_without_chunk
        )
    )
    train, val, stats = select_subset(
        candidates,
        train_seconds=float(subset["train_hours"]) * 3600.0,
        val_seconds=float(subset["val_hours"]) * 3600.0,
        min_speaker_seconds=float(subset["min_speaker_sec"]),
        seed=int(subset["seed"]),
    )
    split_of = {row["id"]: "train" for row in train}
    split_of.update({row["id"]: "val" for row in val})
    by_id = {row["id"]: row for row in list(train) + list(val)}

    rows = []
    for item_id, item in by_id.items():
        audio = dataset[item["source_index"]]["audio"]
        rows.append(
            {
                "id": item["id"],
                "speaker": item["speaker"],
                "split": split_of[item_id],
                "text": item["text"],
                "audio": audio,
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
                "meta": item["meta"],
            }
        )

    def lab_text(text):
        # Keep internal spaces, including a doubled space. Only flatten newlines.
        return text.replace("\n", " ").strip()

    stats.update(
        {
            "dataset": config["hf_dataset"],
            "split_name_on_hub": config["hf_split"],
            "note": (
                "IndicTTS Telugu on the hub is one split of both speakers "
                "(female about 4.27 h, male about 4.47 h, 48 kHz). Neither speaker "
                "alone has 5 hours, so train and validation both contain female and male. "
                "The hub 'train' split is not used as the training set; membership is in this manifest."
            ),
            "selection": {
                "train_hours": subset["train_hours"],
                "val_hours": subset["val_hours"],
                "min_utt_sec": subset["min_utt_sec"],
                "max_utt_sec": subset["max_utt_sec"],
                "min_speaker_sec": subset["min_speaker_sec"],
                "seed": subset["seed"],
                "candidates": len(candidates),
                "indexed_clips": len(catalog),
                "indexed_hours": indexed_seconds / 3600.0,
                "clips_without_usable_slice": clips_without_chunk,
                "chunk_hours": sum(float(item["duration"]) for item in candidates) / 3600.0,
            },
            "chunking_note": (
                "Clips longer than max_utt_sec are split on orthographic syllables. "
                "Audio time is proportional to characters. A slice is dropped when its "
                "transcript is longer than the wav2vec2/MMS CTC frame budget "
                "(the failure mode log_probs 140 vs targets 239). "
                "Female hub clips are often 16 kHz paragraphs; male clips are often 48 kHz. "
                "split_stats train_hours_written and val_hours_written are the real durations. "
                "If reached_target is false, do not describe the subset as 5 h / 1 h."
            ),
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
