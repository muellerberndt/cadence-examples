"""Synthesize MIDI and inspect the actual waveform before accepting an audible revision."""

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audio_metrics(path):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        if width != 2:
            raise ValueError("Expected 16-bit PCM synthesis")
        data = (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float)
            / 32768
        )
    mono = data.reshape(-1, channels).mean(1)
    n = 2048
    blocks = mono[: len(mono) // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    spectrum = np.abs(np.fft.rfft(blocks * np.hanning(n), axis=1))
    flux = np.maximum(0, np.diff(spectrum, axis=0)).mean(axis=1)
    pitch = np.fft.rfftfreq(n, 1 / rate)
    keep = (pitch >= 55) & (pitch < 4000)
    classes = np.round(69 + 12 * np.log2(pitch[keep] / 440)).astype(int) % 12
    chroma = np.bincount(classes, weights=spectrum[:, keep].sum(axis=0), minlength=12)
    chroma /= max(1e-12, chroma.sum())
    return {
        "seconds": len(mono) / rate,
        "peak": float(np.max(np.abs(data))),
        "rms": float(np.sqrt(np.mean(data**2))),
        "silent_fraction": float((rms < 0.0005).mean()),
        "clipped_fraction": float((np.abs(data) > 0.999).mean()),
        "onset_flux_mean": float(flux.mean()),
        "pitch_class_energy": chroma.tolist(),
        "measurement": "Rendered PCM waveform; acoustic health and chroma, not a trained perceptual music critic.",
    }


def render(midi, output):
    executable = shutil.which("fluidsynth")
    font = ROOT / "data/soundfont/GeneralUser-GS.sf2"
    if not executable or not font.exists():
        raise RuntimeError(
            "Install FluidSynth and acquire the documented GeneralUser GS soundfont before synthesis."
        )
    raw = Path(output).with_suffix(".raw.wav")
    subprocess.run(
        [
            executable,
            "-ni",
            "-g",
            ".65",
            "-r",
            "22050",
            "-F",
            str(raw),
            str(font),
            str(midi),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Release tails are real, but the requested export is exactly one minute.
    with wave.open(str(raw), "rb") as inp:
        params = inp.getparams()
        frames = inp.readframes(60 * params.framerate)
    with wave.open(str(output), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    raw.unlink()
    return audio_metrics(output)
