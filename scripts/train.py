#!/usr/bin/env python3
"""Train one FastSpeech 2 model. Resumes from the newest checkpoint unless --from-scratch."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FastSpeech 2 for English or Telugu. "
        "Checkpoints are {step}.pth.tar under English/checkpoints or Telugu/checkpoints. "
        "A killed Kaggle session resumes from the newest step."
    )
    parser.add_argument("--lang", choices=["english", "telugu"], default=None)
    parser.add_argument("--restore-step", type=int, default=None, help="Override the step to resume from.")
    parser.add_argument("--from-scratch", action="store_true", help="Ignore checkpoints and start at step 0.")
    parser.add_argument(
        "--extra-ckpt-dir",
        default=None,
        help="Also search this directory (for example a previous Kaggle output mounted under /kaggle/input).",
    )
    return parser.parse_args()


def newest_checkpoint(directories):
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
        return 0, None
    return max(found, key=lambda item: item[0])


def main():
    args = parse_args()
    from config_resolve import FS2_ROOT, REPO_ROOT, fs2_env, load_config, resolve_lang, write_upstream_yamls

    lang = resolve_lang(args.lang)
    config = load_config(lang)
    preprocessed = REPO_ROOT / config["paths"]["preprocessed"]
    if not (preprocessed / "train.txt").is_file():
        raise SystemExit("Missing {}. Run align and extract_features first.".format(preprocessed / "train.txt"))
    extra_symbols = preprocessed / "extra_symbols.json"
    if not extra_symbols.is_file():
        raise SystemExit("Missing {}.".format(extra_symbols))

    checkpoints = REPO_ROOT / config["paths"]["ckpt"]
    checkpoints.mkdir(parents=True, exist_ok=True)
    extra = args.extra_ckpt_dir or os_environ_extra()
    if args.from_scratch:
        step, source = 0, None
    elif args.restore_step is not None:
        step = args.restore_step
        source = checkpoints / "{}.pth.tar".format(step)
        if not source.is_file() and extra:
            source = Path(extra) / "{}.pth.tar".format(step)
        if step > 0 and not source.is_file():
            raise SystemExit("Checkpoint not found for step {}.".format(step))
    else:
        step, source = newest_checkpoint([checkpoints, extra])

    if source is not None and step > 0:
        destination = checkpoints / "{}.pth.tar".format(step)
        if source.resolve() != destination.resolve():
            print("Copying checkpoint into {}: {}".format(checkpoints, source))
            shutil.copy2(source, destination)

    resolved = REPO_ROOT / "output" / "resolved" / lang
    yamls = write_upstream_yamls(config, resolved)
    print("Training {} from step {} toward {}.".format(lang, step, config["step"]["total_step"]))
    env = fs2_env(extra_symbols)
    command = [
        sys.executable,
        str(FS2_ROOT / "train.py"),
        "--restore_step",
        str(step),
        "-p",
        str(yamls["preprocess.yaml"]),
        "-m",
        str(yamls["model.yaml"]),
        "-t",
        str(yamls["train.yaml"]),
    ]
    subprocess.run(command, cwd=str(FS2_ROOT), env=env, check=True)


def os_environ_extra():
    import os

    return os.environ.get("FS2_EXTRA_CKPT")


if __name__ == "__main__":
    main()
