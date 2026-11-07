#!/usr/bin/env python3
"""One-step CPU/GPU smoke test on synthetic sine waves.

This does not download Svarah or IndicTTS and does not produce submission
metrics. It checks that feature extraction and a single optimizer step run.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic one-step FastSpeech 2 smoke test.")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary corpus.")
    return parser.parse_args()


def _write_sine(path, seconds, sample_rate):
    import numpy as np
    import soundfile as sf

    times = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    wave = 0.6 * np.sin(2 * math.pi * 150 * times) + 0.2 * np.sin(2 * math.pi * 300 * times)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sample_rate, subtype="FLOAT")


def main():
    args = parse_args()
    import tempfile

    import yaml

    from config_resolve import FS2_ROOT, fs2_env
    from textgrid_io import write_textgrid

    root = Path(tempfile.mkdtemp(prefix="fs2-smoke-"))
    sample_rate = 22050
    hop = 256
    raw = root / "raw"
    pre = root / "pre"
    ckpt = root / "ckpt"
    speakers = ["spk_a", "spk_b"]
    manifest = root / "manifest.jsonl"
    rows = []
    # 8 train + 2 val so batch 2 and group 2 fit.
    for index in range(10):
        speaker = speakers[index % 2]
        basename = "utt{:02d}".format(index)
        split = "val" if index < 2 else "train"
        wav = raw / speaker / "{}.wav".format(basename)
        _write_sine(wav, 1.0, sample_rate)
        (raw / speaker / "{}.lab".format(basename)).write_text("hello\n", encoding="utf-8")
        (raw / speaker / "{}.txt".format(basename)).write_text("hello\n", encoding="utf-8")
        write_textgrid(
            pre / "TextGrid" / speaker / "{}.TextGrid".format(basename),
            [("HH", 0.0, 0.25), ("AH0", 0.25, 0.5), ("L", 0.5, 0.75), ("OW", 0.75, 1.0)],
            1.0,
        )
        rows.append({"basename": basename, "speaker": speaker, "split": split})
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    fs2 = str(FS2_ROOT)
    if fs2 not in sys.path:
        sys.path.insert(0, fs2)
    from preprocessor.preprocessor import Preprocessor

    preprocess = {
        "dataset": "Smoke",
        "path": {
            "corpus_path": str(raw),
            "lexicon_path": "",
            "raw_path": str(raw),
            "preprocessed_path": str(pre),
            "split_manifest": str(manifest),
        },
        "preprocessing": {
            "val_size": 0,
            "text": {"text_cleaners": ["english_cleaners"], "language": "en"},
            "audio": {"sampling_rate": sample_rate, "max_wav_value": 32768.0},
            "stft": {"filter_length": 1024, "hop_length": hop, "win_length": 1024},
            "mel": {"n_mel_channels": 80, "mel_fmin": 0, "mel_fmax": 8000},
            "pitch": {"feature": "phoneme_level", "normalization": True},
            "energy": {"feature": "phoneme_level", "normalization": True},
        },
    }
    Preprocessor(preprocess).build_from_path()
    extra = pre / "extra_symbols.json"
    extra.write_text("[]\n", encoding="utf-8")
    model = {
        "transformer": {
            "encoder_layer": 2,
            "encoder_head": 2,
            "encoder_hidden": 64,
            "decoder_layer": 2,
            "decoder_head": 2,
            "decoder_hidden": 64,
            "conv_filter_size": 128,
            "conv_kernel_size": [9, 1],
            "encoder_dropout": 0.0,
            "decoder_dropout": 0.0,
        },
        "variance_predictor": {"filter_size": 64, "kernel_size": 3, "dropout": 0.0},
        "variance_embedding": {
            "pitch_quantization": "linear",
            "energy_quantization": "linear",
            "n_bins": 32,
        },
        "multi_speaker": True,
        "max_seq_len": 200,
        "vocoder": {"model": "HiFi-GAN", "speaker": "universal"},
    }
    train = {
        "path": {
            "ckpt_path": str(ckpt),
            "log_path": str(root / "log"),
            "result_path": str(root / "result"),
        },
        "optimizer": {
            "batch_size": 2,
            "betas": [0.9, 0.98],
            "eps": 1e-9,
            "weight_decay": 0.0,
            "grad_clip_thresh": 1.0,
            "grad_acc_step": 1,
            "warm_up_step": 2,
            "anneal_steps": [100],
            "anneal_rate": 0.5,
            "group_size": 2,
        },
        "step": {
            "total_step": 1,
            "log_step": 1,
            "synth_step": 100,
            "val_step": 1,
            "save_step": 1,
        },
    }
    resolved = root / "yaml"
    resolved.mkdir()
    paths = {}
    for name, payload in ("preprocess.yaml", preprocess), ("model.yaml", model), ("train.yaml", train):
        path = resolved / name
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        paths[name] = path
    env = fs2_env(extra)
    subprocess.run(
        [
            sys.executable,
            str(FS2_ROOT / "train.py"),
            "--restore_step",
            "0",
            "-p",
            str(paths["preprocess.yaml"]),
            "-m",
            str(paths["model.yaml"]),
            "-t",
            str(paths["train.yaml"]),
        ],
        cwd=str(FS2_ROOT),
        env=env,
        check=True,
    )
    checkpoint = ckpt / "1.pth.tar"
    if not checkpoint.is_file():
        raise SystemExit("Smoke test did not write {}".format(checkpoint))
    print("Smoke test passed: {}".format(checkpoint))
    if not args.keep:
        shutil.rmtree(root)
    else:
        print("Kept {}".format(root))


if __name__ == "__main__":
    main()
