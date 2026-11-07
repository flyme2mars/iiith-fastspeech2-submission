"""Write the selected subset to raw_data and a JSONL manifest."""

import json
from collections import Counter
from pathlib import Path

from hf_audio import read_mono, resample, write_wav


def write_subset(rows, raw_root, manifest_path, stats_path, target_rate, lab_text, extra_stats):
    """Decode ``rows`` and write wav/lab/txt plus the split manifest.

    Each row needs: id, speaker, split, text, audio (HF dict), meta (dict).
    ``lab_text(text)`` is the line Montreal Forced Aligner / the aligner reads.
    The original line is stored in ``.txt`` and becomes FastSpeech 2 ``raw_text``.
    """
    raw_root = Path(raw_root)
    manifest_path = Path(manifest_path)
    stats_path = Path(stats_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    source_rates = Counter()
    skipped_silent = 0
    for row in rows:
        waveform, source_rate = read_mono(row["audio"])
        source_rates[int(source_rate)] += 1
        if row.get("start_sec") is not None or row.get("end_sec") is not None:
            start = 0 if row.get("start_sec") is None else int(round(float(row["start_sec"]) * source_rate))
            end = len(waveform) if row.get("end_sec") is None else int(round(float(row["end_sec"]) * source_rate))
            start = max(0, min(start, len(waveform)))
            end = max(start, min(end, len(waveform)))
            waveform = waveform[start:end]
        waveform = resample(waveform, source_rate, target_rate)
        speaker_dir = raw_root / row["speaker"]
        wav_path = speaker_dir / "{}.wav".format(row["id"])
        if not write_wav(wav_path, waveform, target_rate):
            skipped_silent += 1
            continue
        lab = lab_text(row["text"])
        (speaker_dir / "{}.lab".format(row["id"])).write_text(lab + "\n", encoding="utf-8")
        (speaker_dir / "{}.txt".format(row["id"])).write_text(row["text"].strip() + "\n", encoding="utf-8")
        duration = float(len(waveform) / target_rate) if target_rate else 0.0
        written.append(
            {
                "id": row["id"],
                "basename": row["id"],
                "speaker": row["speaker"],
                "split": row["split"],
                "text": row["text"].strip(),
                "duration_sec": duration,
                "source_sampling_rate": int(source_rate),
                "meta": row.get("meta") or {},
            }
        )

    with open(manifest_path, "w", encoding="utf-8") as handle:
        for row in written:
            payload = {
                "basename": row["basename"],
                "speaker": row["speaker"],
                "split": row["split"],
                "text": row["text"],
                "duration_sec": row["duration_sec"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def seconds(split):
        return sum(row["duration_sec"] for row in written if row["split"] == split)

    stats = dict(extra_stats)
    stats.update(
        {
            "written_utterances": len(written),
            "skipped_silent": skipped_silent,
            "train_seconds_written": seconds("train"),
            "val_seconds_written": seconds("val"),
            "train_hours_written": seconds("train") / 3600.0,
            "val_hours_written": seconds("val") / 3600.0,
            "source_sampling_rates": {str(rate): count for rate, count in sorted(source_rates.items())},
            "target_sampling_rate": target_rate,
            "manifest": str(manifest_path),
            "raw_root": str(raw_root),
        }
    )
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "Wrote {} utterances. Train {:.2f} h, val {:.2f} h. Silent skips: {}.".format(
            len(written),
            stats["train_hours_written"],
            stats["val_hours_written"],
            skipped_silent,
        )
    )
    if not stats.get("reached_target", True):
        print(
            "WARNING: the filtered corpus is shorter than the 5 h / 1 h target. "
            "split_stats.json has the actual durations. Do not describe this as a full 5 h / 1 h subset."
        )
    return stats
