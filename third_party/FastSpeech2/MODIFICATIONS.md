# Modifications to ming024/FastSpeech2

The assignment says to start from this repository and to document every change, especially Telugu.

## What was not forked into a new architecture

The encoder, variance adaptor (duration, pitch, energy), decoder, and Post-Net are unchanged. The loss is still mel + post-net mel + pitch + energy + duration. The optimizer is still Adam with the Noam schedule (`d_model ** -0.5`) and the anneal steps in the language config.

## Files changed

| File | Why |
| --- | --- |
| `audio/stft.py` | `librosa.filters.mel` and `pad_center` keyword changes in librosa 0.10. STFT no longer calls `.cuda()` unconditionally, so preprocessing runs on CPU and on Kaggle. |
| `utils/model.py` | `torch.load(..., weights_only=False)` for PyTorch 2.4+. A missing HiFi-GAN file warns and returns `None` instead of crashing training; synthesis still refuses to run without it. |
| `utils/tools.py` | `to_device` casts energy to float32. It was the only feature left as float64, and CUDA `total_loss.backward()` then raised `Found dtype Double but expected Float`. |
| `dataset.py` | Mel, pitch, and energy are loaded as float32. Metadata split is `split("|", 3)` and UTF-8, so a transcript can contain extra pipes and Telugu. |
| `model/loss.py` | Mel, pitch, energy, and log-duration targets are cast to the prediction dtype before the reduction, so a double target cannot promote the sum. |
| `model/modules.py` | Pitch and energy bins are float32, and bucketize targets are cast to float32. |
| `text/symbols.py` | Loads `FS2_EXTRA_SYMBOLS` (JSON list) so Telugu syllables can be added without editing this list by hand. Order is frozen once training starts. |
| `preprocessor/preprocessor.py` | Respects `path.split_manifest` (no random 512-utterance validation cut). Speaker ids are sorted, so they do not depend on filesystem order. `librosa.load(..., sr=)` is keyword-safe. Duration sums that disagree slightly with the mel length are fitted; larger mismatches are skipped. Original text is read from `.txt` when present. If every validation (or training) TextGrid was missing, a sixth of the surviving utterances is moved so `train.txt` and `val.txt` are both non-empty. Normalized pitch and energy are written as float32. |
| `train.py` | `group_size` shrinks when the subset is smaller than `batch × 4`. Training stops at `total_step` on resume. Losses are appended to `losses.csv`. The last step is always checkpointed, and `latest.txt` stores that step. Waveform logging is skipped when the vocoder is absent. |
| `evaluate.py` | Does not log audio when the vocoder returned `None`. |

## Language changes that live outside this tree

Telugu is not a cleaner flag. ming024 ships English (ARPAbet + CMU/librispeech lexicon + g2p-en) and Mandarin (pinyin). Telugu support is in `scripts/text_frontend.py`:

- Orthographic syllables (consonant + virama sequences and combining marks), not Unidecode and not pinyin.
- Each space character is an `sp` phone, so the doubled space in test sentence T3 is two pause tokens.
- Those tokens are written into the TextGrid `phones` tier. `symbols.py` then sees them as `@` + syllable via `extra_symbols.json`.

English stays on the ming024 ARPAbet path (`lexicon/librispeech-lexicon.txt` + g2p-en). Test sentence E3's doubled space is marked `<sp>` only at synthesis time (`preserve_repeated_spaces`). Training labs are ordinary `english_cleaners` text.

Alignment is not the precomputed LJSpeech TextGrid archive. `scripts/align.py` writes TextGrids:

- Default: CTC forced alignment (`torchaudio.functional.forced_align`). English uses `facebook/wav2vec2-base-960h` and projects character spans onto ARPAbet inside each word (vowels get a longer share of the word). Telugu uses `facebook/mms-1b-fl102` with the `tel` adapter and pools character spans into syllables. A target sequence longer than the emission (for example 239 targets and 140 frames) is skipped. That used to raise `RuntimeError` and stop the run before validation TextGrids were written.
- Optional for English: Montreal Forced Aligner with `english_us_arpa` and the vendored lexicon (`--aligner mfa`). There is no Telugu MFA acoustic model in this scaffold.

Pitch is still pyworld DIO + stonemask. Energy is still the STFT magnitude norm. Both are averaged per phone when the config says `phoneme_level`, then standard-scaled, matching ming024.

Both languages set `multi_speaker: true` and `vocoder.speaker: universal`. Svarah has no single 5-hour speaker, and neither IndicTTS Telugu speaker has 5 hours alone.
