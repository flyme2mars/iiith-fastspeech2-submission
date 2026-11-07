"""Deterministic ~5 h train / ~1 h validation selection.

Utterances are filtered by the caller (duration, empty text). This module
then drops speakers who do not have enough audio for a speaker embedding,
splits each remaining speaker so validation is about one sixth of that
speaker, and trims each side to the requested budget. Train and validation
never share an utterance id.
"""

import random


def ensure_both_splits(train, val, val_fraction=1.0 / 6.0):
    """Keep both sides non-empty when at least two utterances exist.

    The per-speaker holdout never puts a speaker's only utterance in
    validation, so two speakers with one clip each used to yield an empty
    val pool. The same thing happens if every validation clip is later
    dropped (short audio whose transcript cannot be CTC-aligned). Move a
    deterministic tail of the surviving pool, about one sixth, onto the
    empty side. Membership stays disjoint.
    """
    train = list(train)
    val = list(val)
    if train and val:
        return train, val
    pool = train + val
    if len(pool) < 2:
        return train, val
    n_val = int(round(len(pool) * float(val_fraction)))
    n_val = max(1, min(n_val, len(pool) - 1))
    # One side was empty, so `pool` is just the surviving utterances.
    return pool[:-n_val], pool[-n_val:]


def take_budget(rows, budget, rng):
    """Round-robin across speakers until ``budget`` seconds are reached."""
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["speaker"], []).append(dict(row))
    speakers = sorted(by_speaker)
    for speaker in speakers:
        rng.shuffle(by_speaker[speaker])

    chosen = []
    total = 0.0
    for speaker in speakers:
        if total >= budget or not by_speaker[speaker]:
            break
        utterance = by_speaker[speaker].pop()
        chosen.append(utterance)
        total += float(utterance["duration"])

    while total < budget:
        progressed = False
        for speaker in speakers:
            if total >= budget or not by_speaker[speaker]:
                continue
            utterance = by_speaker[speaker].pop()
            chosen.append(utterance)
            total += float(utterance["duration"])
            progressed = True
        if not progressed:
            break
    return chosen, total


def select_subset(items, train_seconds, val_seconds, min_speaker_seconds, seed):
    if train_seconds <= 0 or val_seconds <= 0:
        raise ValueError("Train and validation budgets must be positive.")
    by_speaker = {}
    for item in items:
        by_speaker.setdefault(item["speaker"], []).append(dict(item))

    kept = {}
    dropped = []
    for speaker, utterances in by_speaker.items():
        total = sum(float(item["duration"]) for item in utterances)
        if total < min_speaker_seconds:
            dropped.append(speaker)
            continue
        kept[speaker] = utterances
    if not kept:
        raise RuntimeError(
            "No speaker has at least {} seconds after filtering.".format(min_speaker_seconds)
        )

    rng = random.Random(seed)
    fraction = val_seconds / (train_seconds + val_seconds)
    train_pool = []
    val_pool = []
    for speaker in sorted(kept):
        utterances = list(kept[speaker])
        rng.shuffle(utterances)
        target_val = sum(float(item["duration"]) for item in utterances) * fraction
        val_duration = 0.0
        for index, utterance in enumerate(utterances):
            remaining = len(utterances) - index - 1
            # Keep at least one training utterance when the speaker has two or more.
            if val_duration < target_val and remaining >= 1:
                val_pool.append(utterance)
                val_duration += float(utterance["duration"])
            else:
                train_pool.append(utterance)

    val_backfilled = False
    if not train_pool or not val_pool:
        train_pool, val_pool = ensure_both_splits(train_pool, val_pool)
        val_backfilled = True

    train, train_seconds_actual = take_budget(train_pool, train_seconds, rng)
    val, val_seconds_actual = take_budget(val_pool, val_seconds, rng)
    if not train or not val:
        train, val = ensure_both_splits(train, val)
        train_seconds_actual = sum(float(item["duration"]) for item in train)
        val_seconds_actual = sum(float(item["duration"]) for item in val)
        val_backfilled = True

    train_ids = {item["id"] for item in train}
    val_ids = {item["id"] for item in val}
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError("Train/validation overlap: {}".format(sorted(overlap)[:5]))

    def speaker_seconds(rows):
        totals = {}
        for row in rows:
            totals[row["speaker"]] = totals.get(row["speaker"], 0.0) + float(row["duration"])
        return totals

    train_by_speaker = speaker_seconds(train)
    recommended = None
    if train_by_speaker:
        recommended = max(train_by_speaker, key=train_by_speaker.get)

    stats = {
        "seed": seed,
        "target_train_seconds": train_seconds,
        "target_val_seconds": val_seconds,
        "train_seconds": train_seconds_actual,
        "val_seconds": val_seconds_actual,
        "train_utterances": len(train),
        "val_utterances": len(val),
        "speakers_kept": sorted(kept),
        "speakers_dropped_short": sorted(dropped),
        "train_seconds_by_speaker": train_by_speaker,
        "val_seconds_by_speaker": speaker_seconds(val),
        "recommended_speaker": recommended,
        "val_backfilled": val_backfilled,
        "reached_target": (
            train_seconds_actual >= 0.95 * train_seconds
            and val_seconds_actual >= 0.95 * val_seconds
        ),
    }
    return train, val, stats
