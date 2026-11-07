"""Write Praat TextGrids that ming024's preprocessor can read back with tgt."""

from pathlib import Path


def contiguous_intervals(intervals, total):
    """Fill gaps with ``sil`` and make the tier cover ``[0, total]``."""
    if total <= 0:
        raise ValueError("Audio duration must be positive.")
    if not intervals:
        return [("sil", 0.0, float(total))]

    fixed = []
    cursor = 0.0
    for phone, start, end in intervals:
        start = max(float(start), cursor)
        end = max(float(end), start + 0.001)
        if start > cursor + 0.0005:
            fixed.append(("sil", cursor, start))
        fixed.append((str(phone), start, end))
        cursor = end

    if cursor > total + 1e-4:
        scale = float(total) / cursor
        fixed = [(phone, start * scale, end * scale) for phone, start, end in fixed]
        cursor = float(total)
    if cursor < total:
        fixed.append(("sil", cursor, float(total)))

    cleaned = []
    for phone, start, end in fixed:
        if end - start < 1e-4:
            continue
        if cleaned and abs(start - cleaned[-1][2]) > 1e-4:
            cleaned.append(("sil", cleaned[-1][2], start))
        cleaned.append((phone, start, end))
    if not cleaned:
        return [("sil", 0.0, float(total))]
    phone, start, _end = cleaned[-1]
    cleaned[-1] = (phone, start, float(total))
    return cleaned


def write_textgrid(path, intervals, total):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tier = contiguous_intervals(intervals, total)
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        "xmax = {:.6f}".format(total),
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "phones"',
        "        xmin = 0",
        "        xmax = {:.6f}".format(total),
        "        intervals: size = {}".format(len(tier)),
    ]
    for index, (phone, start, end) in enumerate(tier, start=1):
        safe = str(phone).replace('"', "'")
        lines.extend(
            [
                "        intervals [{}]:".format(index),
                "            xmin = {:.6f}".format(start),
                "            xmax = {:.6f}".format(end),
                '            text = "{}"'.format(safe),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tier
