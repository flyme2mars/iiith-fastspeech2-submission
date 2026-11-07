"""CTC length budget shared by the Telugu subsetter and the aligner.

torchaudio's forced_align raises RuntimeError when the target sequence is
longer than the emission (``L_frames >= L_targets + N_repeats``). A single
pair like 140 frames versus 239 targets used to abort the whole Kaggle run.
Callers treat a failed check as "skip this utterance".
"""

import unicodedata

# facebook/wav2vec2 and MMS convolutional front-end.
_CONV = ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2))


def ctc_frames(duration_sec, sample_rate=16000):
    """Emission length of a waveform of ``duration_sec`` after the wav2vec2 CNN."""
    samples = int(float(duration_sec) * int(sample_rate))
    if samples <= 0:
        return 0
    length = samples
    for kernel, stride in _CONV:
        length = (length - kernel) // stride + 1
        if length <= 0:
            return 0
    return int(length)


def ctc_target_length(text):
    """Characters plus one extra frame for every consecutive repeat.

    MMS Telugu is a character CTC vocabulary: one id per code point, and a
    repeated character needs a blank between the two copies. This is the
    same inequality forced_align enforces. It can over-drop a pair whose
    tokenizer is coarser than characters; those still align if they reach
    the model. It must not under-drop the 239-target / 140-frame failures.
    """
    chars = list(unicodedata.normalize("NFC", "" if text is None else str(text).strip()))
    if not chars:
        return 0
    repeats = sum(1 for index in range(1, len(chars)) if chars[index] == chars[index - 1])
    return len(chars) + repeats


def ctc_pair_ok(duration_sec, text, sample_rate=16000):
    needed = ctc_target_length(text)
    if needed <= 0:
        return False
    return ctc_frames(duration_sec, sample_rate=sample_rate) >= needed


def assert_alignable_lengths(n_frames, target_ids):
    """Raise AlignmentError instead of letting forced_align raise RuntimeError."""
    from aligners import AlignmentError

    ids = [int(item) for item in target_ids]
    if not ids:
        raise AlignmentError("empty CTC targets")
    repeats = sum(1 for index in range(1, len(ids)) if ids[index] == ids[index - 1])
    needed = len(ids) + repeats
    if int(n_frames) < needed:
        raise AlignmentError(
            "targets length too long for CTC (log_probs length {}, targets length {})".format(
                int(n_frames), len(ids)
            )
        )
