#!/usr/bin/env python3
"""Synthesize the assignment test sentences and save variance-adaptor arrays.

Audio goes to Results/<Language>/test_generated_audios/.
Prediction arrays go to variance_adaptor_plots/_predictions/.
Run scripts/plot_variance.py afterwards (this script calls it when synthesis succeeds).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SENTENCE_FILES = {
    "english": "data/test_sentences/english.json",
    "telugu": "data/test_sentences/telugu.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize every assignment test sentence with the trained checkpoint."
    )
    parser.add_argument("--lang", choices=["english", "telugu"], default=None)
    parser.add_argument("--restore-step", type=int, default=None)
    parser.add_argument("--speaker-id", type=int, default=None)
    parser.add_argument(
        "--extra-ckpt-dir",
        default=None,
        help="Directory to search in addition to the language checkpoints folder.",
    )
    return parser.parse_args()


def _newest(directories):
    found = []
    for directory in directories:
        if not directory:
            continue
        folder = Path(directory)
        if not folder.is_dir():
            continue
        for path in folder.glob("*.pth.tar"):
            stem = path.name.split(".")[0]
            if stem.isdigit():
                found.append((int(stem), path))
    if not found:
        return None
    return max(found, key=lambda item: item[0])


def _speaker_id(preprocessed, stats_path, override):
    speakers = json.loads((preprocessed / "speakers.json").read_text(encoding="utf-8"))
    if override is not None:
        names = [name for name, index in speakers.items() if int(index) == int(override)]
        if not names:
            raise SystemExit("speaker id {} is not in {}".format(override, speakers))
        return int(override), names[0]
    if not stats_path.is_file():
        raise SystemExit("Missing {} and no --speaker-id was given.".format(stats_path))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    name = stats.get("recommended_speaker")
    if name not in speakers:
        raise SystemExit("recommended speaker {!r} is not in speakers.json.".format(name))
    return int(speakers[name]), name


def _phones_for(lang, text, config, repo_root):
    from text_frontend import english_phones, load_lexicon, tokenize_telugu

    if lang == "telugu":
        return tokenize_telugu(text)
    fs2 = str(repo_root / "third_party" / "FastSpeech2")
    if fs2 not in sys.path:
        sys.path.insert(0, fs2)
    from text.cleaners import english_cleaners
    from text_frontend import ensure_g2p

    lexicon = load_lexicon(repo_root / config["paths"]["lexicon"])
    phones, _words = english_phones(
        text,
        lexicon,
        english_cleaners,
        g2p=ensure_g2p(),
        preserve_repeated_spaces=bool(config["text"].get("preserve_repeated_spaces", False)),
    )
    return phones


def main():
    args = parse_args()
    import os

    from config_resolve import FS2_ROOT, REPO_ROOT, fs2_env, load_config, resolve_lang

    lang = resolve_lang(args.lang)
    config = load_config(lang)
    preprocessed = REPO_ROOT / config["paths"]["preprocessed"]
    extra_symbols = preprocessed / "extra_symbols.json"
    stats_path = (REPO_ROOT / config["paths"]["manifest"]).parent / "split_stats.json"
    if not extra_symbols.is_file() or not (preprocessed / "stats.json").is_file():
        raise SystemExit("Preprocessed features are missing under {}.".format(preprocessed))

    checkpoints = REPO_ROOT / config["paths"]["ckpt"]
    extra = args.extra_ckpt_dir or os.environ.get("FS2_EXTRA_CKPT")
    found = _newest([checkpoints, extra])
    if args.restore_step is not None:
        step = args.restore_step
        source = checkpoints / "{}.pth.tar".format(step)
        if not source.is_file() and extra:
            source = Path(extra) / "{}.pth.tar".format(step)
    elif found is None:
        step, source = None, None
    else:
        step, source = found
    if step is None or source is None or not Path(source).is_file():
        raise SystemExit(
            "No checkpoint in {} (or --extra-ckpt-dir). Train before synthesizing. "
            "Refusing to write silent wavs.".format(checkpoints)
        )
    if Path(source).resolve() != (checkpoints / "{}.pth.tar".format(step)).resolve():
        import shutil

        checkpoints.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, checkpoints / "{}.pth.tar".format(step))

    vocoder_path = FS2_ROOT / "hifigan" / "generator_universal.pth.tar"
    if not vocoder_path.is_file():
        raise SystemExit("Missing {}. Run scripts/download_vocoder.py.".format(vocoder_path))

    sentences = json.loads((REPO_ROOT / SENTENCE_FILES[lang]).read_text(encoding="utf-8"))
    speaker_id, speaker_name = _speaker_id(
        preprocessed,
        stats_path,
        args.speaker_id if args.speaker_id is not None else config["inference"].get("speaker_id"),
    )
    print("Using checkpoint step {} and speaker {} (id {}).".format(step, speaker_name, speaker_id))

    os.environ["FS2_EXTRA_SYMBOLS"] = str(extra_symbols)
    if str(FS2_ROOT) not in sys.path:
        sys.path.insert(0, str(FS2_ROOT))

    import numpy as np
    import torch

    from text import text_to_sequence
    from utils.model import get_model, get_vocoder, vocoder_infer
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from config_resolve import upstream_configs

    preprocess, model_config, train_config = upstream_configs(config)

    class Restore:
        restore_step = step

    # get_model and get_vocoder resolve files relative to the FastSpeech 2 tree.
    previous = os.getcwd()
    os.chdir(FS2_ROOT)
    try:
        model = get_model(Restore, (preprocess, model_config, train_config), device, train=False)
        vocoder = get_vocoder(model_config, device)
    finally:
        os.chdir(previous)
    if vocoder is None:
        raise SystemExit("HiFi-GAN failed to load from {}.".format(vocoder_path))

    norm = json.loads((preprocessed / "stats.json").read_text(encoding="utf-8"))
    _pitch_min, _pitch_max, pitch_mean, pitch_std = norm["pitch"]
    _energy_min, _energy_max, energy_mean, energy_std = norm["energy"]
    cleaners = config["text"]["text_cleaners"]
    audio_dir = REPO_ROOT / config["paths"]["result"] / "test_generated_audios"
    prediction_dir = (
        REPO_ROOT / config["paths"]["result"] / "variance_adaptor_plots" / "_predictions"
    )
    audio_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    sample_rate = int(config["audio"]["sampling_rate"])
    hop = int(config["audio"]["hop_length"])
    for sentence in sentences:
        phones = _phones_for(lang, sentence["text"], config, REPO_ROOT)
        curly = "{" + " ".join(phones) + "}"
        sequence = np.array(text_to_sequence(curly, cleaners), dtype=np.int64)
        if sequence.size != len(phones):
            raise SystemExit(
                "Symbol lookup dropped phones for {}. Check extra_symbols.json.".format(sentence["id"])
            )
        texts = sequence[np.newaxis, :]
        speakers = np.array([speaker_id])
        src_lens = np.array([sequence.size])
        batch = (
            [sentence["id"]],
            [sentence["text"]],
            speakers,
            texts,
            src_lens,
            int(sequence.size),
        )
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(*(batch[2:]))
        mel_len = int(output[9][0].item())
        src_len = int(output[8][0].item())
        postnet = output[1][0, :mel_len].detach().transpose(0, 1).unsqueeze(0)
        duration = output[5][0, :src_len].detach().cpu().numpy()
        pitch = output[2][0, :src_len].detach().cpu().numpy() * pitch_std + pitch_mean
        energy = output[3][0, :src_len].detach().cpu().numpy() * energy_std + energy_mean
        wavs = vocoder_infer(
            postnet,
            vocoder,
            model_config,
            preprocess,
            lengths=np.array([mel_len * hop]),
        )
        wav_path = audio_dir / "{}.wav".format(sentence["id"])
        sf.write(str(wav_path), wavs[0], sample_rate)
        (audio_dir / "{}.phones.txt".format(sentence["id"])).write_text(
            curly + "\n" + sentence["text"] + "\n",
            encoding="utf-8",
        )
        np.savez(
            prediction_dir / "{}.npz".format(sentence["id"]),
            sentence_id=sentence["id"],
            phones=np.array(phones, dtype=object),
            duration=duration,
            pitch=pitch,
            energy=energy,
            raw_text=sentence["text"],
            speaker=speaker_name,
            restore_step=step,
        )
        print("Wrote {}".format(wav_path))

    from plot_variance import plot_sentence

    output_root = REPO_ROOT / config["paths"]["result"] / "variance_adaptor_plots"
    for path in sorted(prediction_dir.glob("*.npz")):
        plot_sentence(path, output_root)
    print("Synthesis finished for {}.".format(lang))


if __name__ == "__main__":
    main()
