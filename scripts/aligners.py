"""Forced alignment backends.

CTC is the Kaggle default: Montreal Forced Aligner is the aligner ming024
documents, but a Telugu acoustic model is not part of that setup and MFA's
Kaldi install often does not finish inside a fresh T4 session. English can
still be aligned with MFA via ``--aligner mfa``.

CTC durations are real alignments, then projected onto the FastSpeech 2 phone
sequence (ARPAbet for English, orthographic syllables for Telugu). Utterances
that do not line up are skipped and counted. Nothing is filled with a flat
duration if alignment fails.
"""

import shutil
import subprocess
from pathlib import Path


class AlignmentError(RuntimeError):
    pass


def align_with_mfa(raw_dir, output_dir, dictionary, acoustic_model):
    if shutil.which("mfa") is None:
        raise SystemExit(
            "Montreal Forced Aligner is not on PATH.\n"
            "Install it in a conda env (recommended by MFA) or re-run with --aligner ctc.\n"
            "English acoustic model used by this repo: {}.\n"
            "Telugu has no MFA acoustic model wired up; use --aligner ctc for Telugu.".format(
                acoustic_model
            )
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "mfa",
        "align",
        str(raw_dir),
        str(dictionary),
        acoustic_model,
        str(output_dir),
        "--clean",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def _load_ctc(model_id, lang_code, token, device):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    kwargs = {}
    if token:
        kwargs["token"] = token
    if lang_code:
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, **kwargs)
        processor.tokenizer.set_target_lang(lang_code)
        model.load_adapter(lang_code)
    else:
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return processor, model


def _ctc_pieces(waveform_16k, text, processor, model, device):
    """Return list of (piece, start_sec, end_sec) from torchaudio forced alignment."""
    import torch
    import torchaudio

    if not hasattr(torchaudio.functional, "forced_align"):
        raise AlignmentError(
            "torchaudio.functional.forced_align is missing. Use the Kaggle torchaudio "
            "or torchaudio>=2.2. Do not reinstall torch from requirements.txt there."
        )
    numpy_wave = waveform_16k.squeeze().detach().cpu().numpy()
    inputs = processor(numpy_wave, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
    log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().unsqueeze(0)

    encoded = processor.tokenizer(text, return_tensors="pt").input_ids[0]
    special = set()
    for attribute in ("pad_token_id", "bos_token_id", "eos_token_id", "cls_token_id"):
        value = getattr(processor.tokenizer, attribute, None)
        if value is not None:
            special.add(int(value))
    unk = getattr(processor.tokenizer, "unk_token_id", None)
    if unk is not None and any(int(item) == int(unk) for item in encoded):
        raise AlignmentError("tokenizer produced <unk>")
    kept = [int(item) for item in encoded if int(item) not in special]
    if not kept:
        raise AlignmentError("empty CTC targets")
    from ctc_budget import assert_alignable_lengths

    n_frames = int(log_probs.shape[1])
    assert_alignable_lengths(n_frames, kept)
    targets = torch.tensor([kept], dtype=torch.long)
    blank = int(processor.tokenizer.pad_token_id or 0)
    if blank in kept:
        raise AlignmentError("CTC targets contain the blank id {}".format(blank))
    try:
        aligned, scores = torchaudio.functional.forced_align(log_probs, targets, blank=blank)
    except (RuntimeError, ValueError) as exc:
        raise AlignmentError("forced_align failed: {}".format(exc)) from exc
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=blank)
    n_frames = log_probs.shape[1]
    seconds = float(waveform_16k.shape[-1]) / 16000.0
    scale = seconds / max(n_frames, 1)
    pieces = []
    for span in spans:
        piece = processor.tokenizer.convert_ids_to_tokens(int(span.token))
        piece = str(piece).replace("▁", " ").replace("|", "|")
        pieces.append((piece, float(span.start) * scale, float(span.end) * scale))
    return pieces


def _load_waveform(path, device):
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.to("cpu"), 16000


def align_english_utterance(wav_path, word_phone_pairs, processor, model, device):
    """Project character CTC spans onto the ARPAbet sequence.

    ``word_phone_pairs`` is ``[(word, [phones]), ...]`` from the same cleaned
    transcript that is joined into the CTC text. ``<sp>`` is not expected in
    training labs; it is accepted for the repeated-space test path.
    """
    from text_frontend import group_pieces_into_words, split_span

    words = [word for word, _phones in word_phone_pairs if word != "<sp>"]
    align_text = " ".join(words)
    waveform, _sample_rate = _load_waveform(wav_path, device)
    pieces = _ctc_pieces(waveform, align_text.upper(), processor, model, device)
    grouped = group_pieces_into_words(pieces)
    if len(grouped) != len(words):
        raise AlignmentError(
            "word count {} != aligned {}".format(len(words), len(grouped))
        )
    grouped_iter = iter(grouped)
    intervals = []
    for word, phones in word_phone_pairs:
        if word == "<sp>":
            if intervals:
                previous_phone, start, end = intervals[-1]
                cut = min(0.04, max(0.01, (end - start) * 0.2))
                intervals[-1] = (previous_phone, start, end - cut)
                intervals.append(("sp", end - cut, end))
            else:
                intervals.append(("sp", 0.0, 0.04))
            continue
        _text, start, end = next(grouped_iter)
        intervals.extend(split_span(start, end, phones))
    return intervals, float(waveform.shape[-1]) / 16000.0


def align_telugu_utterance(wav_path, tokens, processor, model, device):
    """Align a Telugu syllable sequence.

    The CTC model sees the surface string (spaces included). Its character
    spans are then pooled into the tokenizer's syllables. A length mismatch
    skips the file instead of inventing durations.
    """
    from text_frontend import detokenize_telugu

    surface = detokenize_telugu(tokens)
    waveform, _sample_rate = _load_waveform(wav_path, device)
    pieces = _ctc_pieces(waveform, surface, processor, model, device)
    # Collapse model pieces into a character tape, treating separator tokens as spaces.
    tape = []
    for piece, start, end in pieces:
        if piece in {"|", "[PAD]", "<pad>"}:
            chars = [" "]
        else:
            chars = list(piece)
        if not chars:
            continue
        width = max(end - start, 1e-4) / len(chars)
        for offset, char in enumerate(chars):
            tape.append((char, start + offset * width, start + (offset + 1) * width))
    surface_chars = list(surface)
    if len(tape) != len(surface_chars):
        raise AlignmentError(
            "character count {} != aligned {}".format(len(surface_chars), len(tape))
        )
    intervals = []
    cursor = 0
    for token in tokens:
        if token == "sp":
            piece = " "
            width = 1
        else:
            piece = token
            width = len(token)
        chunk = tape[cursor : cursor + width]
        cursor += width
        if len(chunk) != width:
            raise AlignmentError("syllable mapping ran out of characters")
        intervals.append((token, chunk[0][1], chunk[-1][2]))
    if cursor != len(tape):
        raise AlignmentError("unused aligned characters")
    return intervals, float(waveform.shape[-1]) / 16000.0


def load_aligner(language, model_id, lang_code, token):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code = lang_code if language == "te" else None
    processor, model = _load_ctc(model_id, code, token, device)
    return processor, model, device
