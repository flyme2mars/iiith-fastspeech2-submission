#!/usr/bin/env python3
"""Build TextGrids (duration targets) for one language."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align raw_data wav/lab pairs into Praat TextGrids. "
        "Default aligner is CTC (Kaggle). English can use Montreal Forced Aligner."
    )
    parser.add_argument("--lang", choices=["english", "telugu"], default=None)
    parser.add_argument("--aligner", choices=["ctc", "mfa"], default=None)
    parser.add_argument("--limit", type=int, default=0, help="Align at most this many utterances.")
    parser.add_argument(
        "--max-skip-fraction",
        type=float,
        default=None,
        help="Fail if more than this fraction of utterances cannot be aligned.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Realign even when a TextGrid already exists.",
    )
    return parser.parse_args()


def _jobs(raw_root, manifest_path):
    jobs = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            wav = raw_root / row["speaker"] / "{}.wav".format(row["basename"])
            lab = raw_root / row["speaker"] / "{}.lab".format(row["basename"])
            jobs.append((row, wav, lab))
    return jobs


def main():
    args = parse_args()
    from config_resolve import REPO_ROOT, load_config, resolve_lang

    lang = resolve_lang(args.lang)
    config = load_config(lang)
    aligner = args.aligner or config["align"]["default"]
    raw_root = REPO_ROOT / config["paths"]["raw"]
    preprocessed = REPO_ROOT / config["paths"]["preprocessed"]
    manifest = REPO_ROOT / config["paths"]["manifest"]
    if not manifest.is_file():
        script = "download_svarah.py" if lang == "english" else "download_indictts.py"
        raise SystemExit("Missing {}. Run scripts/{} first.".format(manifest, script))

    max_skip = (
        float(args.max_skip_fraction)
        if args.max_skip_fraction is not None
        else float(config["align"]["max_skip_fraction"])
    )
    jobs = _jobs(raw_root, manifest)
    if args.limit:
        jobs = jobs[: args.limit]
    textgrid_root = preprocessed / "TextGrid"
    if aligner == "mfa":
        if lang != "english":
            raise SystemExit(
                "MFA is wired for English (english_us_arpa + the librispeech lexicon). "
                "Telugu has no acoustic model in this scaffold; use --aligner ctc."
            )
        from aligners import align_with_mfa

        dictionary = REPO_ROOT / config["align"]["mfa_dictionary"]
        align_with_mfa(
            raw_root,
            textgrid_root,
            dictionary,
            config["align"]["mfa_acoustic_model"],
        )
        print("MFA wrote TextGrids under {}".format(textgrid_root))
        return

    from tqdm import tqdm

    from aligners import AlignmentError, align_english_utterance, align_telugu_utterance, load_aligner
    from hf_audio import read_hf_token
    from text_frontend import english_word_groups, load_lexicon, tokenize_telugu
    from textgrid_io import write_textgrid

    token = read_hf_token()
    processor, model, device = load_aligner(
        config["language"],
        config["align"]["ctc_model"],
        config["align"].get("ctc_lang"),
        token,
    )
    if lang == "english":
        fs2 = str(REPO_ROOT / "third_party" / "FastSpeech2")
        if fs2 not in sys.path:
            sys.path.insert(0, fs2)
        from text.cleaners import english_cleaners
        from text_frontend import ensure_g2p

        lexicon = load_lexicon(REPO_ROOT / config["paths"]["lexicon"])
        g2p = ensure_g2p()

    failures = []
    done = 0
    reused = 0
    for row, wav, lab in tqdm(jobs, desc="align {}".format(lang)):
        destination = textgrid_root / row["speaker"] / "{}.TextGrid".format(row["basename"])
        if destination.is_file() and destination.stat().st_size > 0 and not args.force:
            done += 1
            reused += 1
            continue
        try:
            transcript = lab.read_text(encoding="utf-8").strip()
            if lang == "english":
                groups = english_word_groups(
                    transcript,
                    lexicon,
                    english_cleaners,
                    g2p=g2p,
                    preserve_repeated_spaces=False,
                )
                intervals, total = align_english_utterance(wav, groups, processor, model, device)
            else:
                tokens = tokenize_telugu(transcript)
                intervals, total = align_telugu_utterance(wav, tokens, processor, model, device)
            write_textgrid(destination, intervals, total)
            done += 1
        except (AlignmentError, KeyError, ValueError, OSError, RuntimeError) as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            failures.append(
                {
                    "basename": row["basename"],
                    "split": row.get("split"),
                    "error": message,
                }
            )

    failed_by_split = {}
    for item in failures:
        failed_by_split[item.get("split") or "?"] = failed_by_split.get(item.get("split") or "?", 0) + 1
    report = {
        "lang": lang,
        "aligner": "ctc",
        "model": config["align"]["ctc_model"],
        "attempted": len(jobs),
        "aligned": done,
        "reused_existing": reused,
        "failed": len(failures),
        "failed_by_split": failed_by_split,
        "failures": failures[:50],
    }
    preprocessed.mkdir(parents=True, exist_ok=True)
    report_path = preprocessed / "align_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Aligned {}/{} utterances. Report: {}".format(done, len(jobs), report_path))
    if not jobs:
        raise SystemExit("Manifest produced no utterances to align.")
    if len(failures) / len(jobs) > max_skip:
        raise SystemExit(
            "Skip fraction {:.2%} exceeds {:.2%}. See {}.".format(
                len(failures) / len(jobs), max_skip, report_path
            )
        )


if __name__ == "__main__":
    main()
