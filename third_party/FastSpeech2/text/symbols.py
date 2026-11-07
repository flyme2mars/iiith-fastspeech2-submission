""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

import json
import os

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Symbols that ship with ming024. Telugu aksharas (and any unseen phone) are
# appended from FS2_EXTRA_SYMBOLS, a JSON list written during feature extraction.
# The file order is the embedding order, so do not regenerate it after training.
_base_symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)


def _extra_symbols():
    path = os.environ.get("FS2_EXTRA_SYMBOLS", "")
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        raise RuntimeError("FS2_EXTRA_SYMBOLS must be a JSON list: {}".format(path))
    return [str(item) for item in loaded]


_known = set(_base_symbols)
_extras = []
for _symbol in _extra_symbols():
    if _symbol not in _known:
        _extras.append(_symbol)
        _known.add(_symbol)

symbols = list(_base_symbols) + _extras
