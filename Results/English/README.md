# English results

Files from the finished English FastSpeech 2 run. Telugu results are not in this folder.

- `training_loss.png`: training loss curve (`scripts/plot_losses.py --lang english`)
- `validation_loss.png`: validation loss curve (same script)
- `test_generated_audios/E1.wav`, `E2a.wav`, `E2b.wav`, `E3.wav`, `E4.wav`: synthesized test sentences (`English/inference.py`)
- `test_generated_audios/E1.phones.txt`, `E2a.phones.txt`, `E2b.phones.txt`, `E3.phones.txt`, `E4.phones.txt`: phone string and sentence text written beside each wav
- `variance_adaptor_plots/duration/{E1,E2a,E2b,E3,E4}.png`
- `variance_adaptor_plots/pitch/{E1,E2a,E2b,E3,E4}.png`
- `variance_adaptor_plots/energy/{E1,E2a,E2b,E3,E4}.png`

`variance_adaptor_plots/_predictions/` holds the numpy arrays those plots read (`*.npz`). Those archives are gitignored. They are not a substitute for the pngs.
