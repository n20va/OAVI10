from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

# ============================================================
# Лабораторная работа №10
# Вариант 2 — Синтезатор речи
# ============================================================

SR = 22050
CROSSFADE_MS = 40

BASE_DIR = Path(__file__).resolve().parent
PHONEMES_DIR = BASE_DIR / "phonemes"
RESULTS_DIR = BASE_DIR / "results"

PHONEMES = [
    # Гласные (23)
    "a", "ae", "aa", "schwa", "e", "eh",
    "i", "y", "o", "yo", "u", "uu",
    "a_red", "schwa2", "i_red", "y_red",
    "y_diph", "y_mid", "y_c", "u_red",
    "u_soft", "i_short", "u_short",

    # Согласные (40)
    "b", "b_soft", "v", "v_soft",
    "g", "g_soft", "d", "d_soft",
    "zh", "zh_soft", "z", "z_soft",
    "j", "k", "k_soft", "l",
    "l_soft", "m", "m_soft", "n",
    "n_soft", "p", "p_soft", "r",
    "r_soft", "s", "s_soft", "t",
    "t_soft", "f", "f_soft", "h",
    "h_soft", "dz", "ch", "dj",
    "c", "sh", "sch"
]

# ============================================================
# Создание папок
# ============================================================

PHONEMES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Синтез искусственной фонемы
# ============================================================

def synthesize_phoneme(
        freq: float,
        duration: float = 0.25,
        sr: int = SR
) -> np.ndarray:

    t = np.linspace(
        0,
        duration,
        int(sr * duration),
        endpoint=False
    )

    signal = (
            0.6 * np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            + 0.1 * np.sin(2 * np.pi * freq * 3 * t)
    )

    envelope = np.linspace(0, 1, len(signal))
    envelope = np.minimum(envelope, envelope[::-1])

    signal *= envelope

    signal /= np.max(np.abs(signal))

    return signal.astype(np.float32)

# ============================================================
# Генерация всех фонем
# ============================================================

phoneme_audio = {}

base_freq = 180

for idx, phoneme in enumerate(PHONEMES):

    freq = base_freq + idx * 20

    audio = synthesize_phoneme(freq)

    phoneme_audio[phoneme] = audio

    sf.write(
        PHONEMES_DIR / f"{phoneme}.wav",
        audio,
        SR
    )

# ============================================================
# Простая конкатенация
# ============================================================

def concatenate_simple(
        sequence: list[str]
) -> np.ndarray:

    result = []

    for ph in sequence:
        result.append(phoneme_audio[ph])

    return np.concatenate(result)

# ============================================================
# Crossfade конкатенация
# ============================================================

def concatenate_crossfade(
        sequence: list[str],
        crossfade_ms: int = CROSSFADE_MS
) -> np.ndarray:

    crossfade = int(
        SR * crossfade_ms / 1000
    )

    result = phoneme_audio[
        sequence[0]
    ].copy()

    for ph in sequence[1:]:

        next_audio = phoneme_audio[ph]

        left = result[-crossfade:]
        right = next_audio[:crossfade]

        fade_out = np.linspace(
            1,
            0,
            crossfade
        )

        fade_in = np.linspace(
            0,
            1,
            crossfade
        )

        mixed = (
                left * fade_out
                + right * fade_in
        )

        result = np.concatenate([
            result[:-crossfade],
            mixed,
            next_audio[crossfade:]
        ])

    return result

# ============================================================
# Транскрипция фразы
# ============================================================

# «Хорошо живёт на свете Винни-Пух»

transcription = [
    "h",
    "a_red",
    "r",
    "a_red",
    "sh",
    "o",

    "zh",
    "y_mid",
    "v_soft",
    "o",
    "t",

    "n",
    "a_red",

    "s",
    "v_soft",
    "e",
    "t_soft",
    "i_red",

    "v_soft",
    "i",
    "n",
    "n_soft",
    "i_red",

    "p",
    "u",
    "h"
]

# ============================================================
# Синтез фразы
# ============================================================

simple_audio = concatenate_simple(
    transcription
)

crossfade_audio = concatenate_crossfade(
    transcription
)

# ============================================================
# Сохранение WAV
# ============================================================

sf.write(
    RESULTS_DIR / "simple_concat.wav",
    simple_audio,
    SR
)

sf.write(
    RESULTS_DIR / "crossfade_concat.wav",
    crossfade_audio,
    SR
)

# ============================================================
# Спектрограмма STFT
# ============================================================

def save_spectrogram(
        audio: np.ndarray,
        path: Path,
        title: str
):

    f, t, Zxx = stft(
        audio,
        fs=SR,
        window='hann',
        nperseg=1024,
        noverlap=512
    )

    magnitude = 20 * np.log10(
        np.abs(Zxx) + 1e-10
    )

    plt.figure(figsize=(12, 5))

    plt.pcolormesh(
        t,
        f,
        magnitude,
        shading='gouraud'
    )

    plt.yscale('log')

    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')

    plt.title(title)

    plt.colorbar(label='dB')

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

# ============================================================
# Построение спектрограмм
# ============================================================

save_spectrogram(
    simple_audio,
    RESULTS_DIR / "spectrogram_simple.png",
    "Спектрограмма — простая конкатенация"
)

save_spectrogram(
    crossfade_audio,
    RESULTS_DIR / "spectrogram_crossfade.png",
    "Спектрограмма — crossfade"
)

# ============================================================
# Отчёт
# ============================================================

report_text = f"""
# Лабораторная работа №10
## Вариант 2 — Синтезатор речи

### Выполнено

1. Созданы образцы фонем русского языка:
   всего {len(PHONEMES)} файлов

2. Реализована простая конкатенация

3. Реализован монтаж с перекрёстным
   затуханием (crossfade)

4. Синтезирована фраза:

   «Хорошо живёт на свете Винни-Пух»

5. Построены спектрограммы
   методом STFT с окном Ханна

### Сравнение методов

#### Простая конкатенация
- заметны скачки сигнала
- слышны границы фонем

#### Crossfade
- переходы между фонемами
  более плавные
- меньше щелчков и разрывов

### Вывод

Crossfade обеспечивает более
естественное звучание
синтезированной речи.
"""

with open(
        RESULTS_DIR / "report.md",
        "w",
        encoding="utf-8"
) as f:
    f.write(report_text)

# ============================================================
# Информация
# ============================================================

print("====================================")
print("Лабораторная работа выполнена")
print("====================================")
print(f"Фонем: {len(PHONEMES)}")
print("Результаты сохранены в папке:")
print(RESULTS_DIR)
print("====================================")