"""Stdlib checks for the scaffold. No datasets and no GPU."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chunk_text import chunk_utterance  # noqa: E402
from ctc_budget import assert_alignable_lengths, ctc_frames, ctc_pair_ok  # noqa: E402
from subset import ensure_both_splits, select_subset  # noqa: E402
from text_frontend import (  # noqa: E402
    detokenize_telugu,
    english_phones,
    tokenize_telugu,
)
from textgrid_io import contiguous_intervals, write_textgrid  # noqa: E402

E3 = "The rhyth  m of the orchestra changed unexpectedly."
T3 = "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది."
T1 = "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."
T2A = "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."
T2B = "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."
T4 = "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."


def _dummy_cleaner(text):
    import re

    return re.sub(r"\s+", " ", text.lower()).strip()


class SubsetTests(unittest.TestCase):
    def test_budgets_are_disjoint(self):
        items = []
        for speaker in ("a", "b", "c", "short"):
            count = 2 if speaker == "short" else 12
            duration = 5 if speaker == "short" else 10
            for index in range(count):
                items.append(
                    {
                        "id": "{}_{}".format(speaker, index),
                        "speaker": speaker,
                        "duration": duration,
                    }
                )
        train, val, stats = select_subset(
            items,
            train_seconds=200,
            val_seconds=40,
            min_speaker_seconds=30,
            seed=42,
        )
        train_ids = {row["id"] for row in train}
        val_ids = {row["id"] for row in val}
        self.assertFalse(train_ids & val_ids)
        self.assertLessEqual(stats["train_seconds"], 200 + 10)
        self.assertLessEqual(stats["val_seconds"], 40 + 10)
        self.assertGreater(stats["train_seconds"], 150)
        self.assertGreater(stats["val_seconds"], 30)
        self.assertIn("short", stats["speakers_dropped_short"])
        self.assertNotIn("short", stats["speakers_kept"])
        self.assertTrue(stats["recommended_speaker"])

    def test_same_seed_is_stable(self):
        items = [
            {"id": "u{}".format(index), "speaker": "s", "duration": 10}
            for index in range(30)
        ]
        first, _, _ = select_subset(items, 100, 20, 30, 42)
        second, _, _ = select_subset(items, 100, 20, 30, 42)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])


class TeluguSplitTests(unittest.TestCase):
    def test_ctc_length_mismatch_is_an_alignment_error(self):
        self.assertEqual(ctc_frames(2.81), 140)
        self.assertFalse(ctc_pair_ok(2.81, "క" * 239))
        with self.assertRaises(Exception) as caught:
            assert_alignable_lengths(140, list(range(239)))
        self.assertIn("targets length too long", str(caught.exception))
        assert_alignable_lengths(140, [3, 4, 5])

    def test_long_paragraph_is_sliced_and_short_mismatch_is_dropped(self):
        phrase = "నమస్కారం ఇది ఒక వాక్యం. "
        chunks = chunk_utterance(phrase * 30, 40.0, max_sec=12.0, min_sec=1.0)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertAlmostEqual(sum(chunk["duration"] for chunk in chunks), 40.0, places=5)
        self.assertTrue(all(chunk["duration"] <= 12.0 + 1e-3 for chunk in chunks))
        self.assertTrue(all(ctc_pair_ok(chunk["duration"], chunk["text"]) for chunk in chunks))
        self.assertEqual(chunk_utterance("క" * 239, 2.81, max_sec=12.0, min_sec=1.0), [])

    def test_single_clip_speakers_still_get_validation(self):
        items = [
            {"id": "female_only", "speaker": "female", "duration": 40.0},
            {"id": "male_only", "speaker": "male", "duration": 40.0},
        ]
        train, val, stats = select_subset(items, 5 * 3600, 3600, 30, 42)
        self.assertTrue(train and val)
        self.assertEqual({row["id"] for row in train} & {row["id"] for row in val}, set())
        self.assertEqual(len(train) + len(val), 2)
        self.assertTrue(stats["val_backfilled"])
        self.assertFalse(stats["reached_target"])

    def test_short_unalignable_clips_do_not_consume_the_val_budget(self):
        items = []
        for speaker in ("female", "male"):
            for index in range(40):
                items.append(
                    {
                        "id": "{}_bad_{}".format(speaker, index),
                        "speaker": speaker,
                        "duration": 1.2,
                        "text": "క" * 239,
                    }
                )
            for index in range(8):
                items.append(
                    {
                        "id": "{}_ok_{}".format(speaker, index),
                        "speaker": speaker,
                        "duration": 8.0,
                        "text": "నమస్కారం.",
                    }
                )
        usable = [row for row in items if ctc_pair_ok(row["duration"], row["text"])]
        self.assertTrue(all("_ok_" in row["id"] for row in usable))
        train, val, _stats = select_subset(usable, 5 * 3600, 3600, 30, 42)
        self.assertTrue(train and val)
        self.assertTrue(all("_ok_" in row["id"] for row in train + val))
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in val})

    def test_empty_val_after_dropped_shorts_is_refilled(self):
        train, val = ensure_both_splits(
            [{"id": "t{}".format(index), "duration": 1.2} for index in range(12)],
            [],
        )
        self.assertTrue(train and val)
        self.assertEqual(len(train) + len(val), 12)
        self.assertGreater(len(train), len(val))

    def test_missing_val_textgrids_still_write_both_meta_files(self):
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")
        fs2 = ROOT / "third_party" / "FastSpeech2"
        if str(fs2) not in sys.path:
            sys.path.insert(0, str(fs2))
        from preprocessor.preprocessor import Preprocessor

        from textgrid_io import write_textgrid

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            pre = root / "pre"
            sample_rate = 22050
            manifest = root / "manifest.jsonl"
            rows = []
            for index in range(6):
                split = "val" if index < 2 else "train"
                basename = "utt{:02d}".format(index)
                speaker = "spk_a"
                wav = raw / speaker / "{}.wav".format(basename)
                wav.parent.mkdir(parents=True, exist_ok=True)
                times = np.arange(sample_rate, dtype=np.float32) / sample_rate
                wave = 0.2 * np.sin(2 * np.pi * 180 * times)
                sf.write(str(wav), wave, sample_rate)
                (raw / speaker / "{}.lab".format(basename)).write_text("hello\n", encoding="utf-8")
                if split == "train":
                    write_textgrid(
                        pre / "TextGrid" / speaker / "{}.TextGrid".format(basename),
                        [("HH", 0.0, 0.25), ("AH0", 0.25, 0.5), ("L", 0.5, 0.75), ("OW", 0.75, 1.0)],
                        1.0,
                    )
                rows.append({"basename": basename, "speaker": speaker, "split": split})
            with open(manifest, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            preprocess = {
                "dataset": "Smoke",
                "path": {
                    "corpus_path": str(raw),
                    "lexicon_path": "",
                    "raw_path": str(raw),
                    "preprocessed_path": str(pre),
                    "split_manifest": str(manifest),
                },
                "preprocessing": {
                    "val_size": 0,
                    "text": {"text_cleaners": ["english_cleaners"], "language": "en"},
                    "audio": {"sampling_rate": sample_rate, "max_wav_value": 32768.0},
                    "stft": {"filter_length": 1024, "hop_length": 256, "win_length": 1024},
                    "mel": {"n_mel_channels": 80, "mel_fmin": 0, "mel_fmax": 8000},
                    "pitch": {"feature": "phoneme_level", "normalization": True},
                    "energy": {"feature": "phoneme_level", "normalization": True},
                },
            }
            Preprocessor(preprocess).build_from_path()
            train_lines = [line for line in (pre / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
            val_lines = [line for line in (pre / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(train_lines and val_lines)
            self.assertEqual(len(train_lines) + len(val_lines), 4)


class TextTests(unittest.TestCase):
    def test_english_repeated_space_becomes_sp(self):
        phones, words = english_phones(
            "rhyth  m",
            {"rhyth": ["R", "IH1"], "m": ["M"]},
            _dummy_cleaner,
            preserve_repeated_spaces=True,
        )
        self.assertEqual(phones, ["R", "IH1", "sp", "M"])
        self.assertIn("<sp>", words)

    def test_english_single_space_has_no_sp(self):
        phones, _words = english_phones(
            "rhyth m",
            {"rhyth": ["R"], "m": ["M"]},
            _dummy_cleaner,
            preserve_repeated_spaces=True,
        )
        self.assertEqual(phones, ["R", "M"])

    def test_telugu_roundtrip_and_double_space(self):
        import unicodedata

        for text in (T1, T2A, T2B, T3, T4):
            tokens = tokenize_telugu(text)
            self.assertEqual(
                detokenize_telugu(tokens),
                unicodedata.normalize("NFC", text.strip()),
            )
        tokens = tokenize_telugu(T3)
        self.assertTrue(
            any(tokens[index] == "sp" and tokens[index + 1] == "sp" for index in range(len(tokens) - 1))
        )

    def test_assignment_files_match_pdf_text(self):
        english = json.loads((ROOT / "data/test_sentences/english.json").read_text(encoding="utf-8"))
        telugu = json.loads((ROOT / "data/test_sentences/telugu.json").read_text(encoding="utf-8"))
        by_id = {row["id"]: row["text"] for row in english}
        self.assertEqual(by_id["E3"], E3)
        self.assertEqual(by_id["E3"].count("  "), 1)
        self.assertIn("andaroo", by_id["E4"])
        self.assertIn("10:45", by_id["E4"])
        te = {row["id"]: row["text"] for row in telugu}
        self.assertEqual(te["T1"], T1)
        self.assertEqual(te["T2a"], T2A)
        self.assertEqual(te["T2b"], T2B)
        self.assertEqual(te["T3"], T3)
        self.assertEqual(te["T4"], T4)
        self.assertEqual(len(english), 5)
        self.assertEqual(len(telugu), 5)


class TextGridTests(unittest.TestCase):
    def test_contiguous_cover(self):
        tier = contiguous_intervals([("HH", 0.2, 0.5)], 1.0)
        self.assertEqual(tier[0][0], "sil")
        self.assertAlmostEqual(tier[0][2], 0.2)
        self.assertAlmostEqual(tier[-1][2], 1.0)

    def test_tgt_roundtrip(self):
        try:
            import tgt
        except ImportError:
            self.skipTest("tgt is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utt.TextGrid"
            write_textgrid(path, [("HH", 0.0, 0.4), ("అ", 0.4, 0.9)], 1.0)
            grid = tgt.io.read_textgrid(str(path), encoding="utf-8")
            tier = grid.get_tier_by_name("phones")
            labels = [interval.text for interval in tier._objects]
            self.assertIn("HH", labels)
            self.assertIn("అ", labels)


class ConfigTests(unittest.TestCase):
    def test_language_configs(self):
        import yaml

        for name, language, dataset in (
            ("english.yaml", "en", "Svarah"),
            ("telugu.yaml", "te", "IndicTTS_Telugu"),
        ):
            config = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
            self.assertEqual(config["language"], language)
            self.assertEqual(config["dataset"], dataset)
            self.assertEqual(config["audio"]["sampling_rate"], 22050)
            self.assertEqual(config["audio"]["hop_length"], 256)
            self.assertEqual(config["audio"]["mel_fmax"], 8000)
            self.assertEqual(config["subset"]["train_hours"], 5.0)
            self.assertEqual(config["subset"]["val_hours"], 1.0)
            self.assertTrue(config["model"]["multi_speaker"])
            self.assertEqual(config["model"]["vocoder"]["speaker"], "universal")
            self.assertEqual(config["step"]["total_step"], 20000)
            self.assertEqual(config["step"]["save_step"], 2000)
            self.assertEqual(config["optimizer"]["batch_size"], 16)


if __name__ == "__main__":
    unittest.main()
