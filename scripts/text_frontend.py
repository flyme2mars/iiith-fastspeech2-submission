"""Text frontends for the English and Telugu FastSpeech 2 models.

English training and inference use ARPAbet from the librispeech lexicon, with
g2p-en for out-of-vocabulary words (ming024's LJSpeech path). Telugu has no
lexicon in that repository, so Telugu uses orthographic syllables. Repeated
spaces are preserved as extra ``sp`` phones because the assignment test set
splits "rhythm" and "అడవిలో" on purpose.
"""

import re
import unicodedata

VIRAMA = "\u0C4D"
_WORD = re.compile(r"<sp>|[a-z0-9']+")
_VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}
_STOPS = {"B", "D", "G", "K", "P", "T"}


def mark_repeated_spaces(text):
    """Turn each extra space into an explicit ``<sp>`` token.

    ``"rhyth  m"`` becomes ``"rhyth <sp> m"``. A normal single space is left
    alone so ordinary word boundaries stay ordinary.
    """

    def replace(match):
        extras = len(match.group(0)) - 1
        return " " + " ".join(["<sp>"] * extras) + " "

    return re.sub(r" {2,}", replace, text)


def collapse_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def english_word_groups(text, lexicon, cleaner, g2p=None, preserve_repeated_spaces=False):
    """Return ``[(word, [phones]), ...]`` for an English utterance.

    ``cleaner`` is applied after repeated-space marking. Pass
    ``text.cleaners.english_cleaners`` in the real pipeline. ``lexicon`` maps
    a lower-case word to a list of ARPAbet phones. The word ``<sp>`` marks a
    repeated-space pause and carries the single phone ``sp``.
    """
    marked = mark_repeated_spaces(text) if preserve_repeated_spaces else text
    cleaned = cleaner(marked)
    words = _WORD.findall(cleaned.lower())
    groups = []
    for word in words:
        if word == "<sp>":
            groups.append((word, ["sp"]))
            continue
        if word in lexicon:
            groups.append((word, list(lexicon[word])))
            continue
        if g2p is None:
            raise KeyError(word)
        guessed = []
        for piece in g2p(word):
            phone = str(piece).strip().upper()
            if phone and phone not in {" ", "'"}:
                guessed.append(phone)
        if not guessed:
            raise KeyError(word)
        groups.append((word, guessed))
    if not groups:
        raise ValueError("No English phones produced for: {}".format(text))
    return groups


def english_phones(text, lexicon, cleaner, g2p=None, preserve_repeated_spaces=False):
    """Return (flat phones, words)."""
    groups = english_word_groups(
        text,
        lexicon,
        cleaner,
        g2p=g2p,
        preserve_repeated_spaces=preserve_repeated_spaces,
    )
    phones = [phone for _word, group in groups for phone in group]
    words = [word for word, _group in groups]
    return phones, words


def load_lexicon(path):
    lexicon = {}
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            if word not in lexicon:
                lexicon[word] = parts[1:]
    return lexicon


def is_combining(char):
    if char in ("\u200c", "\u200d"):
        return True
    return unicodedata.category(char).startswith("M")


def tokenize_telugu(text):
    """Split Telugu text into orthographic syllables, one token per space.

    Each ASCII or Telugu space character becomes its own ``sp`` token, so a
    doubled space survives into the duration target and the test-time input.
    """
    tokens = []
    chars = list(unicodedata.normalize("NFC", text.strip()))
    index = 0
    count = len(chars)
    while index < count:
        if chars[index].isspace():
            tokens.append("sp")
            index += 1
            continue
        end = index + 1
        while end < count and not chars[end].isspace():
            previous = chars[end - 1]
            if is_combining(chars[end]) or previous == VIRAMA:
                end += 1
                continue
            break
        tokens.append("".join(chars[index:end]))
        index = end
    if not tokens:
        raise ValueError("No Telugu tokens produced for: {}".format(text))
    return tokens


def detokenize_telugu(tokens):
    return "".join(" " if token == "sp" else token for token in tokens)


def phones_to_curly(phones):
    return "{" + " ".join(phones) + "}"


def phone_weight(phone):
    base = phone.rstrip("012")
    if base in _VOWELS:
        return 1.3
    if base in _STOPS:
        return 0.7
    return 1.0


def split_span(start, end, phones):
    """Divide ``[start, end]`` across phones. Vowels get a longer share."""
    if not phones:
        raise ValueError("split_span requires at least one phone")
    start = float(start)
    end = float(end)
    if end <= start:
        end = start + 0.001 * len(phones)
    weights = [phone_weight(phone) for phone in phones]
    total = sum(weights)
    cursor = start
    spans = []
    width = end - start
    for index, (phone, weight) in enumerate(zip(phones, weights)):
        if index == len(phones) - 1:
            nxt = end
        else:
            nxt = cursor + width * weight / total
        if nxt <= cursor:
            nxt = cursor + 0.001
        spans.append((phone, cursor, nxt))
        cursor = nxt
    return spans


def group_pieces_into_words(pieces):
    """Group CTC pieces into words. ``|`` and spaces are separators."""
    words = []
    buffer = []
    start = None
    end = None

    def flush():
        nonlocal buffer, start, end
        if buffer and start is not None and end is not None and end > start:
            words.append(("".join(buffer), start, end))
        buffer = []
        start = None
        end = None

    for text, piece_start, piece_end in pieces:
        if text in {"|", " ", ""}:
            flush()
            continue
        if start is None:
            start = piece_start
        end = piece_end
        buffer.append(text)
    flush()
    return words


def ensure_g2p():
    """Import g2p-en and download the small NLTK tagger it expects."""
    import nltk

    for resource, lookup in (
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("cmudict", "corpora/cmudict"),
    ):
        try:
            nltk.data.find(lookup)
        except LookupError:
            nltk.download(resource, quiet=True)
    from g2p_en import G2p

    return G2p()
