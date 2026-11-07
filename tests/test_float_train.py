"""One optimizer step must stay float32 when features are saved as float64.

pyworld pitch and preprocessor.normalize used to write float64 energy. CUDA
then raised RuntimeError: Found dtype Double but expected Float inside
total_loss.backward(). This test feeds that dtype through the real loss.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FS2 = ROOT / "third_party" / "FastSpeech2"
sys.path.insert(0, str(FS2))

from model.loss import FastSpeech2Loss  # noqa: E402
from utils.tools import to_device  # noqa: E402


def _loss_module():
    preprocess = {
        "preprocessing": {
            "pitch": {"feature": "phoneme_level"},
            "energy": {"feature": "phoneme_level"},
        }
    }
    return FastSpeech2Loss(preprocess, {})


class FloatTrainTests(unittest.TestCase):
    def test_to_device_casts_float64_energy_and_pitch(self):
        batch = (
            ["utt"],
            ["hello"],
            np.array([0], dtype=np.int64),
            np.array([[1, 2, 3]], dtype=np.int64),
            np.array([3], dtype=np.int64),
            3,
            np.ones((1, 4, 2), dtype=np.float64),
            np.array([4], dtype=np.int64),
            4,
            np.ones((1, 3), dtype=np.float64),
            np.ones((1, 3), dtype=np.float64),
            np.array([[1, 1, 2]], dtype=np.int64),
        )
        moved = to_device(batch, torch.device("cpu"))
        self.assertEqual(moved[6].dtype, torch.float32)
        self.assertEqual(moved[9].dtype, torch.float32)
        self.assertEqual(moved[10].dtype, torch.float32)
        self.assertEqual(moved[11].dtype, torch.long)

    def test_one_backward_with_double_targets_stays_float(self):
        loss_fn = _loss_module()
        mel_pred = torch.randn(2, 4, 5, requires_grad=True)
        post = torch.randn(2, 4, 5, requires_grad=True)
        pitch_pred = torch.randn(2, 3, requires_grad=True)
        energy_pred = torch.randn(2, 3, requires_grad=True)
        log_dur_pred = torch.randn(2, 3, requires_grad=True)
        src_mask = torch.tensor(
            [[False, False, True], [False, False, False]]
        )
        mel_mask = torch.tensor(
            [
                [False, False, False, True],
                [False, False, False, False],
            ]
        )
        inputs = [None] * 6 + [
            torch.randn(2, 4, 5, dtype=torch.float64),
            None,
            None,
            torch.randn(2, 3, dtype=torch.float64),
            torch.randn(2, 3, dtype=torch.float64),
            torch.tensor([[2, 1, 1], [1, 1, 2]], dtype=torch.int64),
        ]
        predictions = (
            mel_pred,
            post,
            pitch_pred,
            energy_pred,
            log_dur_pred,
            None,
            src_mask,
            mel_mask,
            None,
            None,
        )
        total, mel, postnet, pitch, energy, duration = loss_fn(inputs, predictions)
        self.assertEqual(total.dtype, torch.float32)
        for term in (mel, postnet, pitch, energy, duration):
            self.assertEqual(term.dtype, torch.float32)
        total.backward()
        self.assertEqual(mel_pred.grad.dtype, torch.float32)
        self.assertEqual(energy_pred.grad.dtype, torch.float32)
        self.assertEqual(pitch_pred.grad.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
