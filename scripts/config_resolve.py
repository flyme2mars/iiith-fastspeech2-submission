"""Load configs/english.yaml and configs/telugu.yaml into ming024's three dicts."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FS2_ROOT = REPO_ROOT / "third_party" / "FastSpeech2"

LANG_INFO = {
    "english": {
        "config": REPO_ROOT / "configs" / "english.yaml",
        "results": REPO_ROOT / "Results" / "English",
        "checkpoints": REPO_ROOT / "English" / "checkpoints",
    },
    "telugu": {
        "config": REPO_ROOT / "configs" / "telugu.yaml",
        "results": REPO_ROOT / "Results" / "Telugu",
        "checkpoints": REPO_ROOT / "Telugu" / "checkpoints",
    },
}


def resolve_lang(explicit=None):
    lang = explicit or os.environ.get("FS2_LANG")
    if lang not in LANG_INFO:
        raise SystemExit(
            "Pass --lang english or telugu (got {!r}).".format(lang)
        )
    return lang


def load_config(lang):
    import yaml

    path = LANG_INFO[lang]["config"]
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def _abs(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def upstream_configs(config, root=None):
    """Return (preprocess_config, model_config, train_config) with absolute paths."""
    root = REPO_ROOT if root is None else Path(root)
    raw = _abs(root, config["paths"]["raw"])
    preprocessed = _abs(root, config["paths"]["preprocessed"])
    checkpoints = _abs(root, config["paths"]["ckpt"])
    logs = _abs(root, config["paths"]["log"])
    results = _abs(root, config["paths"]["result"])
    lexicon = _abs(root, config["paths"]["lexicon"]) if config["paths"].get("lexicon") else ""
    manifest = _abs(root, config["paths"]["manifest"])

    preprocess = {
        "dataset": config["dataset"],
        "path": {
            "corpus_path": str(raw),
            "lexicon_path": str(lexicon),
            "raw_path": str(raw),
            "preprocessed_path": str(preprocessed),
            "split_manifest": str(manifest),
        },
        "preprocessing": {
            "val_size": 0,
            "text": {
                "text_cleaners": list(config["text"]["text_cleaners"]),
                "language": config["language"],
                "preserve_repeated_spaces": bool(
                    config["text"].get("preserve_repeated_spaces", False)
                ),
                "representation": config["text"]["representation"],
            },
            "audio": {
                "sampling_rate": int(config["audio"]["sampling_rate"]),
                "max_wav_value": float(config["audio"]["max_wav_value"]),
            },
            "stft": {
                "filter_length": int(config["audio"]["filter_length"]),
                "hop_length": int(config["audio"]["hop_length"]),
                "win_length": int(config["audio"]["win_length"]),
            },
            "mel": {
                "n_mel_channels": int(config["audio"]["n_mel_channels"]),
                "mel_fmin": config["audio"]["mel_fmin"],
                "mel_fmax": config["audio"]["mel_fmax"],
            },
            "pitch": {
                "feature": config["pitch"]["feature"],
                "normalization": bool(config["pitch"]["normalization"]),
            },
            "energy": {
                "feature": config["energy"]["feature"],
                "normalization": bool(config["energy"]["normalization"]),
            },
        },
    }
    model = config["model"]
    train = {
        "path": {
            "ckpt_path": str(checkpoints),
            "log_path": str(logs),
            "result_path": str(results),
        },
        "optimizer": config["optimizer"],
        "step": config["step"],
    }
    return preprocess, model, train


def write_upstream_yamls(config, directory):
    import yaml

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    preprocess, model, train = upstream_configs(config)
    paths = {}
    for name, payload in (
        ("preprocess.yaml", preprocess),
        ("model.yaml", model),
        ("train.yaml", train),
    ):
        path = directory / name
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        paths[name] = path
    return paths


def fs2_env(extra_symbols):
    env = os.environ.copy()
    env["FS2_EXTRA_SYMBOLS"] = str(extra_symbols)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(FS2_ROOT) + (os.pathsep + previous if previous else "")
    return env
