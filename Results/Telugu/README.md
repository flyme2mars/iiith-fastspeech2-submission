# Telugu results

Files from the finished Telugu FastSpeech 2 run (IndicTTS Telugu, 5.00 h train / 1.00 h val, 2020 / 417 clips, female and male; see `logs/telugu_indictts/split_stats.json`). Synthesis used checkpoint step **20000** and speaker `female` (id 0). The validation log is lower at step 10000 (2.4912) than at step 20000 (2.5084). English results are not in this folder.

- `training_loss.png`, `validation_loss.png`: `scripts/plot_losses.py --lang telugu`
- `test_generated_audios/T1.wav`, `T2a.wav`, `T2b.wav`, `T3.wav`, `T4.wav`: `scripts/synthesize.py --lang telugu`
- `test_generated_audios/<id>.phones.txt`: syllable string and sentence text beside each wav
- `variance_adaptor_plots/duration|pitch|energy/{T1,T2a,T2b,T3,T4}.png`

## Known limitations (see `test_generated_audios/oov_backoff_log.txt`)

- Syllables that never appeared in training have no symbol. At inference they back off to their known characters: T1 `శ్లే` → `శ ల`.
- The frontend does not verbalize digits. In T4 the numbers `2026`, `12`, `10:45` were dropped, so the wav skips them. Number-to-word normalization is the fix.

`_predictions/` is an intermediate numpy archive (gitignored). The graded files are the pngs next to it.
