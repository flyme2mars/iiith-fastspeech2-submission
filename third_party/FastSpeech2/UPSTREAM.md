# Upstream

Vendored from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
at commit `d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08, "update README and demo page").

That repository is the implementation named in the assignment
("FastSpeech 2 GitHub implementation" → https://github.com/ming024/FastSpeech2).
It follows [FastSpeech 2](https://arxiv.org/abs/2006.04558) (Ren et al.), using
phoneme-level F0 rather than the continuous-wavelet pitch spectrogram of later versions.

Not copied, on purpose:

- `hifigan/generator_*.pth.tar.zip` (about 50 MB each). `scripts/download_vocoder.py` fetches the universal generator from this same commit.
- `demo/` audio and `preprocessed_data/` file lists.
- `index.html`.

License: MIT, Copyright (c) 2020 Chung-Ming Chien. See `LICENSE`.
Modifications for this assignment: `MODIFICATIONS.md`.
