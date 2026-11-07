import json
import os
import random
from collections import Counter

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    """ming024 feature extraction with a fixed train/validation manifest.

    The original class shuffled utterances and cut `val_size` of them off as
    validation. This assignment needs an explicit ~5 h / ~1 h split with no
    overlap, so when `path.split_manifest` is set that file decides membership.
    Speaker ids are assigned by sorted speaker name (stable across machines).
    """

    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.split_manifest = config["path"].get("split_manifest")

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def _load_manifest(self):
        rows = []
        with open(self.split_manifest, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("split") not in {"train", "val"}:
                    raise RuntimeError(
                        "Manifest split must be 'train' or 'val': {}".format(row)
                    )
                rows.append(row)
        if not rows:
            raise RuntimeError("Split manifest is empty: {}".format(self.split_manifest))
        return rows

    def build_from_path(self):
        os.makedirs(os.path.join(self.out_dir, "mel"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

        print("Processing Data ...")
        out = []
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        skip_reasons = Counter()

        if self.split_manifest:
            manifest = self._load_manifest()
            speaker_names = sorted({row["speaker"] for row in manifest})
            jobs = [
                (row["speaker"], row["basename"], row["split"])
                for row in sorted(manifest, key=lambda row: (row["split"], row["basename"]))
            ]
        else:
            speaker_names = sorted(
                name
                for name in os.listdir(self.in_dir)
                if os.path.isdir(os.path.join(self.in_dir, name))
            )
            jobs = []
            for speaker in speaker_names:
                for wav_name in sorted(os.listdir(os.path.join(self.in_dir, speaker))):
                    if wav_name.endswith(".wav"):
                        jobs.append((speaker, wav_name[: -len(".wav")], None))

        speakers = {name: index for index, name in enumerate(speaker_names)}

        for speaker, basename, split in tqdm(jobs):
            tg_path = os.path.join(
                self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
            )
            if not os.path.exists(tg_path):
                skip_reasons["missing_textgrid"] += 1
                continue
            ret = self.process_utterance(speaker, basename)
            if ret is None:
                skip_reasons["process_failed"] += 1
                continue
            info, pitch, energy, n_frame = ret
            if split is not None:
                info = info + "|" + split
            out.append(info)
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))
            n_frames += n_frame

        if not out:
            raise RuntimeError(
                "No utterances were preprocessed. Skip counts: {}".format(
                    dict(skip_reasons)
                )
            )
        if not hasattr(pitch_scaler, "mean_"):
            raise RuntimeError("Pitch statistics are empty. Check pyworld / alignments.")

        print("Computing statistic quantities ...")
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        with open(os.path.join(self.out_dir, "speakers.json"), "w", encoding="utf-8") as handle:
            json.dump(speakers, handle, indent=2, ensure_ascii=False)

        with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as handle:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            json.dump(stats, handle, indent=2)

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )
        if skip_reasons:
            print("Skipped utterances: {}".format(dict(skip_reasons)))

        if self.split_manifest:
            # info was extended with the split tag. Strip it before writing the
            # four-field metadata the FastSpeech 2 dataset reader expects.
            train_rows = []
            val_rows = []
            for row in out:
                meta, split = row.rsplit("|", 1)
                if split == "val":
                    val_rows.append(meta)
                elif split == "train":
                    train_rows.append(meta)
                else:
                    raise RuntimeError("Bad split tag on metadata row: {}".format(split))
        else:
            random.shuffle(out)
            out = [row for row in out if row is not None]
            val_rows = out[: self.val_size]
            train_rows = out[self.val_size :]

        if not train_rows or not val_rows:
            before_train, before_val = len(train_rows), len(val_rows)
            train_rows, val_rows = _ensure_both_splits(train_rows, val_rows)
            if train_rows and val_rows and (len(train_rows), len(val_rows)) != (before_train, before_val):
                print(
                    "One split was empty after feature extraction "
                    "(train {}, val {}). Moved utterances so both files are non-empty "
                    "(train {}, val {}).".format(
                        before_train, before_val, len(train_rows), len(val_rows)
                    )
                )
        self._write_meta("train.txt", train_rows)
        self._write_meta("val.txt", val_rows)
        print("train utterances: {}, val utterances: {}".format(len(train_rows), len(val_rows)))
        if not train_rows or not val_rows:
            raise RuntimeError(
                "Both train.txt and val.txt must be non-empty "
                "(got train {}, val {}).".format(len(train_rows), len(val_rows))
            )
        return out

    def _write_meta(self, name, rows):
        path = os.path.join(self.out_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row + "\n")

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        original_path = os.path.join(self.in_dir, speaker, "{}.txt".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        try:
            textgrid = tgt.io.read_textgrid(tg_path, encoding="utf-8")
        except TypeError:
            textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        if not phone or start >= end:
            return None
        duration = [int(value) for value in duration]
        if sum(duration) <= 0:
            return None

        text = "{" + " ".join(phone) + "}"
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start) : int(self.sampling_rate * end)].astype(
            np.float32
        )
        if wav.size < self.hop_length * 2:
            return None

        if os.path.isfile(original_path):
            with open(original_path, encoding="utf-8") as handle:
                raw_text = handle.readline().strip("\n")
        else:
            with open(text_path, encoding="utf-8") as handle:
                raw_text = handle.readline().strip("\n")
        raw_text = raw_text.replace("|", "/").replace("\n", " ").strip()

        pitch, time_stamp = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, time_stamp, self.sampling_rate)

        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        n_frames = int(mel_spectrogram.shape[1])
        total = int(sum(duration))
        if n_frames <= 1 or total <= 0:
            return None
        if abs(total - n_frames) / max(total, n_frames) > 0.2:
            return None
        duration = _fit_durations(duration, n_frames)
        if sum(duration) != n_frames or any(item < 0 for item in duration):
            return None

        pitch = pitch[:n_frames]
        energy = energy[:n_frames]
        mel_spectrogram = mel_spectrogram[:, :n_frames]
        if np.sum(pitch != 0) <= 1:
            return None

        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))
            pos = 0
            for index, dur in enumerate(duration):
                if dur > 0:
                    pitch[index] = np.mean(pitch[pos : pos + dur])
                else:
                    pitch[index] = 0
                pos += dur
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            pos = 0
            for index, dur in enumerate(duration):
                if dur > 0:
                    energy[index] = np.mean(energy[pos : pos + dur])
                else:
                    energy[index] = 0
                pos += dur
            energy = energy[: len(duration)]

        np.save(
            os.path.join(self.out_dir, "duration", "{}-duration-{}.npy".format(speaker, basename)),
            np.array(duration, dtype=np.int64),
        )
        np.save(
            os.path.join(self.out_dir, "pitch", "{}-pitch-{}.npy".format(speaker, basename)),
            pitch,
        )
        np.save(
            os.path.join(self.out_dir, "energy", "{}-energy-{}.npy".format(speaker, basename)),
            energy,
        )
        np.save(
            os.path.join(self.out_dir, "mel", "{}-mel-{}.npy".format(speaker, basename)),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", ""]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for interval in tier._objects:
            start, end, phone = interval.start_time, interval.end_time, interval.text
            if phones == []:
                if phone in sil_phones:
                    continue
                start_time = start
            if phone not in sil_phones:
                phones.append(phone)
                end_time = end
                end_idx = len(phones)
            else:
                phones.append(phone)
            durations.append(
                int(
                    np.round(end * self.sampling_rate / self.hop_length)
                    - np.round(start * self.sampling_rate / self.hop_length)
                )
            )

        phones = phones[:end_idx]
        durations = durations[:end_idx]
        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        if values.size == 0:
            return values
        lower_q = np.percentile(values, 25)
        upper_q = np.percentile(values, 75)
        lower = lower_q - 1.5 * (upper_q - lower_q)
        upper = upper_q + 1.5 * (upper_q - lower_q)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        std = float(std) if float(std) != 0 else 1.0
        for filename in os.listdir(in_dir):
            path = os.path.join(in_dir, filename)
            values = ((np.load(path).astype(np.float64) - float(mean)) / std).astype(np.float32)
            np.save(path, values)
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))
        return min_value, max_value


def _ensure_both_splits(train_rows, val_rows):
    """Same rule as scripts/subset.py:ensure_both_splits, kept local so this
    file does not import the assignment scripts at feature-extraction time.
    """
    train_rows = list(train_rows)
    val_rows = list(val_rows)
    if train_rows and val_rows:
        return train_rows, val_rows
    pool = train_rows + val_rows
    if len(pool) < 2:
        return train_rows, val_rows
    n_val = max(1, min(len(pool) - 1, int(round(len(pool) / 6.0)) or 1))
    return pool[:-n_val], pool[-n_val:]


def _fit_durations(duration, n_frames):
    duration = [max(0, int(value)) for value in duration]
    if not duration:
        return duration
    total = sum(duration)
    if total > n_frames:
        overflow = total - n_frames
        for index in range(len(duration) - 1, -1, -1):
            take = min(overflow, duration[index])
            duration[index] -= take
            overflow -= take
            if overflow == 0:
                break
    elif total < n_frames:
        duration[-1] += n_frames - total
    return duration
