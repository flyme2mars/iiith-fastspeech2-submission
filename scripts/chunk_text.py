"""Split a long Telugu clip into training slices of at most ``max_sec``.

IndicTTS Telugu female recordings are often whole paragraphs (well over the
12 s training cap). Dropping them leaves too little audio for a 5 h / 1 h
subset. Slices follow orthographic-syllable boundaries, and the audio span
is proportional to the number of characters. A slice whose transcript cannot
fit in the CTC frame budget is omitted rather than sent to forced_align.
"""

from ctc_budget import ctc_pair_ok
from text_frontend import detokenize_telugu, tokenize_telugu


def _token_chars(token):
    return 1 if token == "sp" else len(token)


def chunk_utterance(text, duration, max_sec, min_sec):
    """Return ``[{start, end, duration, text}, ...]`` inside ``[0, duration]``.

    An empty list means the clip has no usable slice (too short, empty, or
    the transcript is longer than the audio can align).
    """
    duration = float(duration)
    text = "" if text is None else str(text).strip()
    max_sec = float(max_sec)
    min_sec = float(min_sec)
    if duration <= 0 or max_sec <= 0 or not text or duration < min_sec:
        return []
    if not ctc_pair_ok(duration, text):
        return []
    if duration <= max_sec:
        return [{"start": 0.0, "end": duration, "duration": duration, "text": text}]

    tokens = tokenize_telugu(text)
    total_chars = sum(_token_chars(token) for token in tokens)
    if total_chars <= 0:
        return []
    max_chars = max(1, int(total_chars * max_sec / duration))
    groups = []
    current = []
    current_chars = 0
    for token in tokens:
        size = _token_chars(token)
        if current and current_chars + size > max_chars:
            groups.append(current)
            current = [token]
            current_chars = size
        else:
            current.append(token)
            current_chars += size
    if current:
        groups.append(current)

    packed = []
    cursor = 0.0
    consumed = 0
    for group in groups:
        size = sum(_token_chars(token) for token in group)
        consumed += size
        end = duration if consumed >= total_chars else duration * (consumed / total_chars)
        piece = detokenize_telugu(group).strip()
        packed.append(
            {
                "start": cursor,
                "end": end,
                "duration": max(0.0, end - cursor),
                "text": piece,
            }
        )
        cursor = end

    merged = []
    for chunk in packed:
        if merged and chunk["duration"] < min_sec and merged[-1]["duration"] + chunk["duration"] <= max_sec + 1e-6:
            combined_text = (merged[-1]["text"] + " " + chunk["text"]).strip()
            combined_duration = chunk["end"] - merged[-1]["start"]
            if combined_text and ctc_pair_ok(combined_duration, combined_text):
                merged[-1]["end"] = chunk["end"]
                merged[-1]["duration"] = combined_duration
                merged[-1]["text"] = combined_text
                continue
        merged.append(chunk)

    kept = []
    for chunk in merged:
        if chunk["duration"] < min_sec or chunk["duration"] > max_sec + 1e-3:
            continue
        if not chunk["text"] or not ctc_pair_ok(chunk["duration"], chunk["text"]):
            continue
        kept.append(chunk)
    return kept
