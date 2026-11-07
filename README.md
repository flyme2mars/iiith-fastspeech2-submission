# FastSpeech 2 for English and Telugu

IIIT-H Speech Processing Lab. Two separate FastSpeech 2 models, for English and Telugu, with the test sentences from the assignment PDF synthesized.

The reference implementation is [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) at commit `d4e79eb` (the link in the assignment). The paper is [FastSpeech 2](https://arxiv.org/abs/2006.04558). What changed, including Telugu, is written in [`third_party/FastSpeech2/MODIFICATIONS.md`](third_party/FastSpeech2/MODIFICATIONS.md).

English and Telugu loss curves, test wavs, phone strings, and variance-adaptor plots are in `Results/English/` and `Results/Telugu/`. The technical report is [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf). Checkpoint files stay gitignored. The step 10000 English weight and the step 20000 Telugu weight are linked in [Checkpoints](#checkpoints).

## How to run this

### 1. Accept Svarah

Svarah is a **gated** Hugging Face dataset. IndicTTS Telugu is public.

1. Log in at https://huggingface.co with the account that will own the token.
2. Open https://huggingface.co/datasets/ai4bharat/Svarah and click **Accept**.
3. Create a read token at https://huggingface.co/settings/tokens.

The token is never written into the repo. Scripts read `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`. If neither is set, they try a Kaggle secret named `HF_TOKEN`.

Telugu does not need a token. The IndicTTS license still applies: https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf

### 2. Put `HF_TOKEN` in Kaggle secrets

1. Create a notebook on https://www.kaggle.com (or upload `notebooks/kaggle_train_english.ipynb`).
2. **Settings → Accelerator → GPU T4**.
3. **Settings → Internet → On** (Hugging Face, the vocoder, and the CTC aligner are downloaded at runtime).
4. **Add-ons → Secrets → Add secret**. Label `HF_TOKEN`, value the read token from step 1. Do not print it.
5. The English notebook loads that secret into the environment. The Telugu notebook loads it if it is present and continues without it.

### 3. Run the two notebooks

Clone this repository inside the notebook (the first code cell does that). On Kaggle the cell checks out `main` and hard-resets to it. The Telugu alignment fix is already on `main`. Locally the cell does not touch git.

Run **English first** (`notebooks/kaggle_train_english.ipynb`), then **Telugu** (`notebooks/kaggle_train_telugu.ipynb`). They are separate models. Do not mix the checkpoints.

Each notebook:

1. Installs `requirements-kaggle.txt` and leaves Kaggle's torch alone.
2. Downloads and subsets the corpus (target ~5 h train / ~1 h val; Telugu writes the real hours to `data/telugu/split_stats.json`).
3. CTC-aligns it and extracts pitch, energy, duration, and mel. An utterance that cannot be aligned is skipped. The run stops only if more than 35% fail.
4. Downloads the universal HiFi-GAN vocoder (~50 MB, not committed).
5. Trains for up to 20,000 steps, writing `{step}.pth.tar` every 2,000 steps under `English/checkpoints` or `Telugu/checkpoints` (both live in `/kaggle/working`, so they survive "Save Version").
6. Plots losses, synthesizes the five test sentences, and writes the variance-adaptor plots.

If the session dies, start the notebook again. `scripts/train.py` resumes from the newest `{step}.pth.tar`. To resume from a previous session's output, add that output as a Kaggle dataset and set `FS2_EXTRA_CKPT` to its checkpoint folder (the train cell has the commented line).

Pitch (pyworld) and the normalized energy arrays are float64 on disk. The loader and the loss cast them to float32 before `backward`. A CUDA run that still has the old code stops at step 1 with `Found dtype Double but expected Float`. No checkpoint is written in that case. After this fix is on `main`, re-run the English notebook from the train cell (the default `restore_step` is 0). Leave `preprocessed_data/english` in place so download and alignment are not repeated. The Kaggle setup cell checks out `main`.

The `Results/` tree and `Report/Technical_Report.pdf` are already committed. Checkpoint files are not. Use the links in [Checkpoints](#checkpoints).

### Local commands

Python 3.10 or newer. From a fresh clone:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_...          # required for Svarah only

python scripts/download_svarah.py
python English/preprocess.py --aligner ctc
python scripts/download_vocoder.py
python English/train.py
python scripts/plot_losses.py --lang english
python English/inference.py

python scripts/download_indictts.py
python Telugu/preprocess.py --aligner ctc
python Telugu/train.py
python scripts/plot_losses.py --lang telugu
python Telugu/inference.py
```

`English/preprocess.py` aligns and extracts. It does not download unless you pass `--download`. Montreal Forced Aligner is optional and English-only:

```bash
python English/preprocess.py --aligner mfa
```

`mfa` must be on `PATH`, with the `english_us_arpa` acoustic model downloaded (`mfa model download acoustic english_us_arpa`). Telugu rejects `--aligner mfa`: this scaffold has no Telugu MFA acoustic model.

Rebuild the report PDF from the `Report/` directory so the `../Results/` image links resolve. wkhtmltopdf is the engine used for the committed PDF. The Lua filter marks Telugu letters for Noto Sans Telugu, because that engine does not fall back per character.

```bash
cd Report
pandoc Technical_Report.md -o Technical_Report.pdf \
  --embed-resources \
  --pdf-engine=wkhtmltopdf \
  --pdf-engine-opt=--enable-local-file-access \
  --lua-filter=telugu-font.lua \
  --metadata title="FastSpeech 2 for English (Svarah) and Telugu (IndicTTS)" \
  -c report.css \
  -f markdown-smart
```

## Layout

```
English/  Telugu/          # PDF entrypoints: train, preprocess, inference, config, checkpoints
configs/                   # the yaml those symlinks point at
scripts/                   # download, align, extract, train, synthesize, plot
third_party/FastSpeech2/   # ming024 source + the patches in MODIFICATIONS.md
data/test_sentences/       # E1 to E4 and T1 to T4, text preserved
notebooks/                 # Kaggle English and Telugu notebooks
Results/English/           # loss pngs, test wavs, phone strings, variance plots
Results/Telugu/            # loss pngs, test wavs, phone strings, OOV log, variance plots
logs/english_svarah/       # Kaggle split_stats and train/val loss logs
logs/telugu_indictts/      # Kaggle split_stats, align_report, train/val loss logs
Report/                    # Technical_Report.md and Technical_Report.pdf
samples/ plots/ checkpoints/   # pointers only
```

`samples/`, `plots/`, and `checkpoints/` are not a second copy of the results.

## Data selection

Both hub datasets are larger than 5 h + 1 h, and neither is already split the way the assignment asks. The download scripts write `data/<lang>/manifest.jsonl` and `split_stats.json` (gitignored). Train and validation utterance ids do not overlap. The seed is 42.

### English: [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)

Svarah is an **accented Indian English ASR benchmark**, not a studio TTS corpus. The hub release is one split named `test`: about **9.6 hours**, **6,656** clips, **117** speakers, read and spontaneous speech, 19 native languages. Columns used here: `audio_filepath`, `duration`, `text`, and the demographic fields (`gender`, `age-group`, `primary_language`, `native_place_state`, `native_place_district`, `highest_qualification`, `job_category`, `occupation_domain`). There is **no `speaker_id`**. The speaker key is a hash of that demographic tuple. Two people who share it are merged. That is a limitation, and `split_stats.json` records how many keys survived.

Filters, then the hour budget:

- Drop empty transcripts.
- Keep clips from **1.0 s to 12.0 s** (short clips have unstable F0; long clips blow the T4 batch).
- Drop speaker keys with under **30 s** total, so the speaker embedding is not a single utterance.
- Within each remaining speaker, hold out about one sixth for validation, then trim globally to **5 h train / 1 h val**, round-robin so speakers stay in both sides when they have the audio.
- The hub split name `test` is not reused as either side. The manifest is the split.

The native sample rate is whatever the files contain (ASR releases of Svarah are commonly 16 kHz). The script records `source_sampling_rates` in `split_stats.json` and resamples to **22050 Hz** for the universal HiFi-GAN. Upsampling does not add bandwidth.

### Telugu: [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)

Studio IndicTTS Telugu, two speakers, hub split `train`, columns `audio`, `text`, `gender`. The card reports **~8.74 h** total (**female ~4.27 h**, **male ~4.47 h**) at **48 kHz**. Neither speaker alone can fill 5 h of training, so the model is multi-speaker and both genders appear in train and validation. The hub split name is not the training split.

Spot checks of the hub preview do not match one 48 kHz rate. Female clips there are **16 kHz** paragraphs, often 20 to 80 s, and male clips are **48 kHz** and usually shorter. `data/telugu/split_stats.json` records `source_sampling_rates` after the subset is written. Treat that file as the measurement. Everything is resampled to 22050 Hz for the vocoder.

A 12 s cap that drops those paragraphs cannot fill 5 h + 1 h. The 12 s value stays as the **training-slice** length so a T4 batch of 16 still fits. `scripts/download_indictts.py` splits longer clips on orthographic syllables and gives each slice a share of the audio proportional to its characters. A slice is discarded when the transcript is longer than the CTC frame budget. That is the pair that crashed Kaggle: `forced_align` raised `RuntimeError` because the targets were longer than the emissions (140 frames versus 239 targets). The exception was not caught, alignment stopped before any validation TextGrid was written, and feature extraction then reported about **3.19 h**, **1744 train / 0 val**, and refused to continue. Alignment now skips that utterance. If one split is still empty after extraction, a sixth of the surviving utterances is moved so `train.txt` and `val.txt` are both non-empty.

The hours that were actually written are `train_hours_written` and `val_hours_written`. If `reached_target` is false, those numbers are the usable subset. The finished logs are in `logs/telugu_indictts/split_stats.json` and `logs/english_svarah/split_stats.json`. Both have `reached_target: true`, 5.00 h train, and 1.00 h val. English is 3505 / 683 clips, all 16 kHz before resampling. Telugu is 2020 / 417 clips, female and male. Rebuild an older Kaggle subset with `python scripts/download_indictts.py --force` (the duration index is reused; add `--reindex` only to scan the hub split again).

The hub README's last paragraph says "Hindi". The dataset card language is `te`, the columns match this Telugu release, and the durations match IndicTTS Telugu. This repo follows the card.

## Model and training budget

| | English | Telugu |
| --- | --- | --- |
| Tokens | ARPAbet (librispeech lexicon + g2p-en) | Orthographic syllables |
| Alignment default | wav2vec2 CTC, projected onto phones | MMS `facebook/mms-1b-fl102` (`tel`) |
| Speakers | multi-speaker embedding | `female` and `male` embeddings |
| Sample rate / hop / mel | 22050 Hz / 256 / 80 mels, fmax 8000 | same |
| Batch / grad accum | 16 / 1 (if the T4 OOMs: batch 8, `grad_acc_step` 2) | same |
| Steps | 20,000, warmup 1,000, anneal at 12k and 16k | same |
| Checkpoint | every 2,000 steps | same |
| Vocoder | HiFi-GAN universal (ming024), not the LJSpeech one | same |

The learning rate is ming024's Noam schedule, base `encoder_hidden ** -0.5` (256 ** -0.5). ming024's LJSpeech recipe is 900,000 steps. 20,000 is the budget aimed at one T4 session on about 5 hours. Both runs reached step 20000. Validation totals from `logs/*/val/log.txt`: English 2.7435 at step 7500, 2.7723 at step 10000, 2.8197 at step 20000. Telugu 2.4912 at step 10000 (the best line) and 2.5084 at step 20000. Training totals are lower at step 20000 than at step 10000 (English 1.5533 vs 2.1000, Telugu 0.9172 vs 1.0532) while validation is higher, which is mild overfitting. The step 10000 checkpoint is the one to synthesize from. The hosted files are still `en/10000.pth.tar` and `te/20000.pth.tar`. `Results/Telugu/README.md` records a step 20000 synthesis and speaker `female` (id 0).

Pitch and energy are phoneme-level and normalized, which is ming024's recommended setting rather than the paper's frame-level unnormalized F0.

## Test sentences the frontend has to survive

The strings in `data/test_sentences/` match the PDF, including internal doubled spaces.

- **E1:** very long word (`antidisestablishmentarianism`).
- **E2:** `conduct` as a noun (a) and a verb (b).
- **E3:** `rhyth  m` (two spaces). Synthesis keeps an extra `sp` phone. Training text does not contain that trick.
- **E4:** `10:45`, `AM`, the date, and the code-switched token `andaroo` (OOV for the lexicon; g2p-en guesses it).
- **T1:** long borrowed technical word.
- **T2:** `పట్టు` as grip (a) and as the silk in `పట్టు చీర` (b).
- **T3:** `అడ  విలో` (two spaces inside అడవిలో). The syllable tokenizer emits two `sp` phones.
- **T4:** date and `10:45` left as digits, plus the odd closing `ఉంచండి`.

`english_cleaners` lowercases, expands numbers, and would collapse the doubled space if we ran it first. Repeated spaces are marked before the cleaner, only when `preserve_repeated_spaces` is on (synthesis). Say so in the report; do not "fix" the sentences.

## Environment

- Python 3.10+.
- PyTorch 2.1+ with a matching torchaudio (`forced_align` needs a recent torchaudio; Kaggle's image has one).
- Kaggle: `pip install -q -r requirements-kaggle.txt`. Do not pip-install torch there.
- Local GPU: install a CUDA torch first if `pip install torch` would fetch a CPU wheel, then `pip install -r requirements.txt`.
- Optional: Montreal Forced Aligner (conda is the reliable install) for English `--aligner mfa`.
- `g2p-en` downloads a small NLTK tagger on first English alignment or synthesis.

`scripts/smoke_test.py` runs one optimizer step on synthetic sines. It is a developer check. It is not a result.

## Checkpoints

Checkpoints and the HiFi-GAN zip are gitignored. The two weights are on Hugging Face:

https://huggingface.co/akshaikrishna/fastspeech2-svarah-indictts

- English, step 10000: https://huggingface.co/akshaikrishna/fastspeech2-svarah-indictts/resolve/main/en/10000.pth.tar
- Telugu, step 20000: https://huggingface.co/akshaikrishna/fastspeech2-svarah-indictts/resolve/main/te/20000.pth.tar

The technical report PDF is [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf). The Markdown source is [`Report/Technical_Report.md`](Report/Technical_Report.md).
