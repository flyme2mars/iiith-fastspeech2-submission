"""Hugging Face token lookup and WAV writing. Never prints the token."""

import io
import os


def read_hf_token():
    """Return a token from the environment or from Kaggle secrets.

    Svarah is gated. The token is the Hugging Face *access token* of an
    account that has clicked Accept on the dataset page. IndicTTS Telugu is
    public; a token is optional there.
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None
    if value and str(value).strip():
        return str(value).strip()
    return None


def require_token(dataset_id):
    token = read_hf_token()
    if token:
        return token
    raise SystemExit(
        "\n".join(
            [
                "No Hugging Face token found.",
                "1. Open https://huggingface.co/datasets/{} and click Accept".format(dataset_id),
                "   while logged into the account that will own the token.",
                "2. Create a read token at https://huggingface.co/settings/tokens",
                "3. Locally:  export HF_TOKEN=hf_...",
                "   On Kaggle: Add-ons -> Secrets -> add a secret named HF_TOKEN,",
                "   then the notebook reads it with kaggle_secrets. Do not commit the token.",
            ]
        )
    )


def audio_header(audio):
    """Return (duration_seconds, sample_rate) without keeping the waveform."""
    import soundfile as sf

    if not isinstance(audio, dict):
        raise TypeError("Expected a Hugging Face audio dict, got {}".format(type(audio)))
    if audio.get("bytes"):
        info = sf.info(io.BytesIO(audio["bytes"]))
        return float(info.duration), int(info.samplerate)
    if audio.get("path"):
        info = sf.info(audio["path"])
        return float(info.duration), int(info.samplerate)
    if audio.get("array") is not None and audio.get("sampling_rate"):
        array = audio["array"]
        sample_rate = int(audio["sampling_rate"])
        return float(len(array) / sample_rate), sample_rate
    raise RuntimeError("Audio dict has no bytes, path, or array.")


def read_mono(audio):
    """Decode an HF audio dict to mono float32 and its sample rate."""
    import numpy as np
    import soundfile as sf

    if audio.get("array") is not None and audio.get("sampling_rate"):
        waveform = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio["sampling_rate"])
    elif audio.get("bytes"):
        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    elif audio.get("path"):
        waveform, sample_rate = sf.read(audio["path"], dtype="float32", always_2d=False)
    else:
        raise RuntimeError("Cannot decode audio dict.")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, int(sample_rate)


def resample(waveform, source_rate, target_rate):
    import librosa
    import numpy as np

    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate == target_rate:
        return waveform
    try:
        return librosa.resample(waveform, orig_sr=source_rate, target_sr=target_rate)
    except TypeError:
        return librosa.resample(waveform, source_rate, target_rate)


def write_wav(path, waveform, sample_rate):
    """Peak-normalize and write 16-bit PCM. Return False for near-silent audio."""
    import numpy as np
    import soundfile as sf

    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak < 1e-5:
        return False
    waveform = waveform / peak
    pcm = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, sample_rate, subtype="PCM_16")
    return True
