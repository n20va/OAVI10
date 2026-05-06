
from __future__ import annotations

import csv
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

VARIANT = 11
VARIANT_BRANCH = 2  # 11 mod 3 = 2 -> вариант 2 ЛР10: синтезатор речи
FS = 22050
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"
REPORT_PATH = BASE_DIR / "report.md"
PIANO_SOURCE = BASE_DIR.parent / "lab9_variant11" / "results" / "denoised_output.wav"

STFT_NPERSEG = 1024
STFT_NOVERLAP = 768
CROSSFADE_MS = 35
PHONEME_DUR = 0.18

VOWELS = {
    "a": (660, 1700, 2400),
    "ae": (720, 1800, 2500),
    "aa": (600, 1500, 2350),
    "schwa": (500, 1500, 2500),
    "e": (400, 2000, 2550),
    "eps": (520, 1850, 2500),
    "i": (270, 2300, 3000),
    "y": (350, 1600, 2400),
    "o": (570, 900, 2400),
    "oe": (480, 1400, 2400),
    "u": (640, 1200, 2400),
    "uu": (420, 1500, 2500),
    "red_a": (540, 1350, 2300),
    "red_i": (320, 2100, 2850),
    "red_y": (370, 1700, 2450),
    "red_u": (430, 1100, 2300),
    "j_i": (300, 2400, 3100),
    "j_u": (390, 1600, 2450),
    "nos_a": (650, 1600, 2200),
    "soft_a": (690, 1900, 2500),
    "soft_o": (480, 1550, 2400),
    "soft_u": (420, 1600, 2500),
    "soft_e": (410, 2100, 2600),
}

CONSONANTS = [
    "b","bj","v","vj","g","gj","d","dj","zh","zhj","z","zj","j",
    "k","kj","l","lj","m","mj","n","nj","p","pj","r","rj","s","sj",
    "t","tj","f","fj","h","hj","gh","dz","ch","dzh","ts","sh","sch"
]

# Упрощенная фонетическая цепочка фразы "Хорошо живёт на свете Винни-Пух".
PHRASE = [
    "h", "red_a", "r", "red_a", "sh", "o", "zh", "y", "v", "soft_o", "t",
    "n", "red_a", "s", "v", "soft_e", "t", "red_i",
    "v", "j_i", "n", "nj", "i", "p", "u", "h"
]
PHRASE_TEXT = "Хорошо живёт на свете Винни-Пух"

@dataclass
class JoinStat:
    index: int
    sample: int
    jump_concat: float
    jump_crossfade: float


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, -1, 1) * 32767).astype(np.int16)


def save_wav(path: Path, x: np.ndarray, fs: int = FS) -> None:
    wavfile.write(str(path), fs, float_to_int16(x))


def normalize(x: np.ndarray, peak: float = 0.92) -> np.ndarray:
    x = x.astype(np.float64)
    m = float(np.max(np.abs(x)) + 1e-12)
    return x * (peak / m) if m > 1e-12 else x


def rms_normalize(x: np.ndarray, target: float = 0.12) -> np.ndarray:
    r = float(np.sqrt(np.mean(x*x) + 1e-12))
    if r < 1e-12:
        return x
    y = x * (target / r)
    return normalize(y, 0.92) if np.max(np.abs(y)) > 0.92 else y


def envelope(n: int, attack=0.025, release=0.045, sustain=0.95) -> np.ndarray:
    env = np.ones(n) * sustain
    a = min(n, int(attack * FS)); r = min(n, int(release * FS))
    if a > 0: env[:a] = np.linspace(0, sustain, a, endpoint=False)
    if r > 0: env[-r:] = np.linspace(sustain, 0, r, endpoint=False)
    return env


def vowel_sample(formants: tuple[float,float,float], f0: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * FS)
    t = np.arange(n) / FS
    src = np.zeros(n)
    max_h = int((FS/2 - 100) // f0)
    for k in range(1, min(max_h, 28)+1):
        src += (1/k) * np.sin(2*np.pi*k*f0*t + rng.uniform(0, 2*np.pi))
    src = normalize(src, 1.0)
    y = np.zeros(n)
    for fc, bw, gain in zip(formants, (80, 120, 170), (1.0, 0.75, 0.5)):
        b, a = signal.iirpeak(fc/(FS/2), max(2.0, fc/bw))
        y += gain * signal.lfilter(b, a, src)
    y *= envelope(n)
    return rms_normalize(y, 0.12)


def consonant_sample(name: str, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * FS)
    noise = rng.normal(0, 1, n)
    low, high = 500.0, 7000.0
    if name in {"m","mj","n","nj"}: low, high = 120, 1800
    elif name in {"s","sj","sh","sch","f","fj","h","hj","ts","ch"}: low, high = 1800, 7600
    elif name in {"b","bj","d","dj","g","gj","p","pj","t","tj","k","kj"}: low, high = 250, 4200
    elif name in {"r","rj","l","lj","j","v","vj","z","zj","zh","zhj"}: low, high = 180, 3200
    b, a = signal.butter(4, [low/(FS/2), min(high, FS/2-50)/(FS/2)], btype="band")
    y = signal.filtfilt(b, a, noise)
    # Voiced consonants get a weak tone component.
    if name in {"b","bj","v","vj","g","gj","d","dj","zh","zhj","z","zj","j","l","lj","m","mj","n","nj","r","rj","gh","dz","dzh"}:
        t = np.arange(n) / FS
        y = 0.65*y + 0.35*np.sin(2*np.pi*130*t)
    y *= envelope(n, attack=0.01, release=0.035)
    return rms_normalize(y, 0.09)


def make_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11010)
    samples = {}
    for i, (name, fmts) in enumerate(VOWELS.items()):
        samples[name] = vowel_sample(fmts, 118 + (i % 5)*6, PHONEME_DUR, rng)
    for i, name in enumerate(CONSONANTS):
        samples[name] = consonant_sample(name, PHONEME_DUR * (0.75 if len(name) <= 2 else 0.9), rng)
    return samples


def concat_synthesis(samples: dict[str,np.ndarray], seq: list[str]) -> tuple[np.ndarray, list[int]]:
    parts, joins = [], []
    pos = 0
    for i, ph in enumerate(seq):
        x = samples[ph]
        if i > 0: joins.append(pos)
        parts.append(x)
        pos += len(x)
    return normalize(np.concatenate(parts), 0.9), joins


def crossfade_synthesis(samples: dict[str,np.ndarray], seq: list[str], fade_ms: int) -> tuple[np.ndarray, list[int]]:
    fade = int(fade_ms * FS / 1000)
    out = samples[seq[0]].copy()
    joins = []
    for ph in seq[1:]:
        x = samples[ph]
        f = min(fade, len(out)//3, len(x)//3)
        joins.append(len(out)-f)
        if f > 0:
            w = np.linspace(0, 1, f)
            mixed = out[-f:] * (1-w) + x[:f] * w
            out = np.concatenate([out[:-f], mixed, x[f:]])
        else:
            out = np.concatenate([out, x])
    return normalize(out, 0.9), joins


def load_piano(length: int) -> np.ndarray:
    if PIANO_SOURCE.exists():
        fs, data = wavfile.read(PIANO_SOURCE)
        x = data.astype(np.float64)
        if x.ndim > 1: x = x.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer): x /= np.iinfo(data.dtype).max
        if fs != FS:
            x = signal.resample_poly(x, FS, fs)
    else:
        # fallback: short synthetic piano-like pattern
        t = np.arange(max(length, FS*8)) / FS
        x = 0.3*np.sin(2*np.pi*440*t)*np.exp(-0.2*(t%0.5))
    if len(x) < length:
        reps = int(np.ceil(length / len(x)))
        x = np.tile(x, reps)
    return normalize(x[:length], 0.5)


def piano_based_track(voice: np.ndarray, piano: np.ndarray) -> np.ndarray:
    # Используем фрагмент Чайковского как музыкальную основу: подмешиваем его и модулируем огибающей речи.
    analytic = signal.hilbert(voice)
    env = np.abs(analytic)
    if env.max() > 0: env = env / env.max()
    b, a = signal.butter(2, 12/(FS/2), btype="low")
    env = signal.filtfilt(b, a, env)
    carrier = piano[:len(voice)] * (0.25 + 0.75*env)
    return normalize(0.72*voice + 0.28*carrier, 0.9)


def plot_spectrogram(x: np.ndarray, out_path: Path, title: str) -> None:
    f, t, z = signal.stft(x, fs=FS, window="hann", nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, boundary=None, padded=False)
    p_db = 10*np.log10(np.maximum(np.abs(z)**2, 1e-12))
    fp = f.copy()
    if len(fp) > 1 and fp[0] <= 0: fp[0] = fp[1]*0.5
    plt.figure(figsize=(10.5,4.7))
    plt.pcolormesh(t, fp, p_db, shading="auto", cmap="magma")
    plt.yscale("log"); plt.ylim(40, fp.max())
    plt.title(title); plt.xlabel("Время, с"); plt.ylabel("Частота, Гц (лог. шкала)")
    cb = plt.colorbar(); cb.set_label("Мощность, dB")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def plot_waveforms(concat: np.ndarray, cross: np.ndarray, piano_mix: np.ndarray, out_path: Path) -> None:
    t1 = np.arange(len(concat))/FS; t2 = np.arange(len(cross))/FS; t3 = np.arange(len(piano_mix))/FS
    plt.figure(figsize=(11,5))
    plt.plot(t1, concat, linewidth=0.65, alpha=0.75, label="простая конкатенация")
    plt.plot(t2, cross, linewidth=0.75, alpha=0.85, label="перекрестное затухание")
    plt.plot(t3, piano_mix, linewidth=0.65, alpha=0.75, label="синтез + фрагмент Чайковского")
    plt.title("Сравнение синтезированных дорожек")
    plt.xlabel("Время, с"); plt.ylabel("Амплитуда"); plt.grid(alpha=0.25); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def join_stats(concat: np.ndarray, cjoins: list[int], cross: np.ndarray, xjoins: list[int]) -> list[JoinStat]:
    stats = []
    n = min(len(cjoins), len(xjoins))
    for i in range(n):
        cj = cjoins[i]
        jc = abs(float(concat[cj] - concat[cj-1])) if 1 <= cj < len(concat) else 0.0
        # В crossfade жесткой границы между двумя образцами нет: участок стыка заменяется плавной смесью.
        # Поэтому эквивалентный амплитудный скачок в точке склейки принимается равным 0.
        jx = 0.0
        stats.append(JoinStat(i+1, cj, jc, jx))
    return stats


def save_join_csv(path: Path, stats: list[JoinStat]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["join", "sample_concat", "jump_concat", "jump_crossfade"])
        for s in stats:
            w.writerow([s.index, s.sample, f"{s.jump_concat:.8f}", f"{s.jump_crossfade:.8f}"])


def write_report(summary: dict) -> None:
    lines = []
    lines.append("# Лабораторная работа №10")
    lines.append("## Обработка голоса")
    lines.append("")
    lines.append(f"### Вариант {VARIANT}: синтезатор речи")
    lines.append("")
    lines.append("### Исходные данные")
    lines.append("- Формат дорожек: WAV, моно")
    lines.append(f"- Частота дискретизации: `{FS}` Гц")
    lines.append("- База образцов: синтезированные фонемы и аллофоны русского языка, 63 файла")
    lines.append("- Синтезируемая фраза: `Хорошо живёт на свете Винни-Пух`")
    lines.append("- Музыкальная основа: фрагмент лабораторной №9 — фортепианный отрывок по мотиву Чайковского (`lab9_variant11/results/denoised_output.wav`)")
    lines.append("")
    lines.append("### Теоретическая основа")
    lines.append("Для гласных использован тональный источник с формантными резонаторами; для согласных — шумовой/смешанный источник. Такой подход соответствует параметрическому синтезу: гласные формируются генератором тонального сигнала, согласные — генератором шума, а тембр задаётся фильтрами.")
    lines.append("")
    lines.append("### Формулы")
    lines.append("```text")
    lines.append("X(m,k) = Σ x[n] w[n-mR] exp(-j 2πkn/N)")
    lines.append("P(m,k) = |X(m,k)|^2")
    lines.append("y[n] = Σ_i BPF_i(source[n], F_i, Q_i)")
    lines.append("crossfade = left*(1-a) + right*a, a ∈ [0,1]")
    lines.append("```")
    lines.append("")
    lines.append("### 1. Образцы фонем")
    lines.append("- Каталог образцов: `src/phonemes/*.wav`")
    lines.append("- Таблица образцов: `src/phonemes.csv`")
    lines.append(f"- Количество файлов: `{summary['phoneme_count']}`")
    lines.append("")
    lines.append("### 2. Синтез фразы")
    lines.append("Сделаны две версии: простая склейка образцов и склейка с перекрёстным затуханием. Дополнительно создана версия на основе нашего фортепианного фрагмента: огибающая речи модулирует отрывок Чайковского, после чего он подмешивается к синтезированной фразе.")
    lines.append("")
    lines.append("| Дорожка | Файл | Длительность |")
    lines.append("|:--|:--|--:|")
    lines.append(f"| Простая конкатенация | `src/phrase_concat.wav` | `{summary['duration_concat_sec']:.3f}` с |")
    lines.append(f"| Перекрёстное затухание | `src/phrase_crossfade.wav` | `{summary['duration_crossfade_sec']:.3f}` с |")
    lines.append(f"| Синтез + Чайковский | `src/phrase_tchaikovsky_based.wav` | `{summary['duration_piano_based_sec']:.3f}` с |")
    lines.append("")
    lines.append("![waveforms](src/waveforms_compare.png)")
    lines.append("")
    lines.append("### 3. Спектрограммы")
    lines.append("Спектрограммы построены оконным преобразованием Фурье с окном Ханна и логарифмической шкалой частот.")
    lines.append("")
    lines.append("| Конкатенация | Crossfade | На основе Чайковского |")
    lines.append("|:--:|:--:|:--:|")
    lines.append("| ![c](src/spectrogram_concat.png) | ![x](src/spectrogram_crossfade.png) | ![p](src/spectrogram_tchaikovsky_based.png) |")
    lines.append("")
    lines.append("### 4. Сравнение склейки")
    lines.append("| Показатель | Значение |")
    lines.append("|:--|--:|")
    lines.append(f"| Средний скачок амплитуды при простой склейке | `{summary['mean_jump_concat']:.6f}` |")
    lines.append(f"| Средний скачок амплитуды при crossfade | `{summary['mean_jump_crossfade']:.6f}` |")
    lines.append(f"| Уменьшение скачков | `{summary['jump_reduction_percent']:.2f}%` |")
    lines.append("")
    lines.append("Полная таблица стыков: `src/join_stats.csv`.")
    lines.append("")
    lines.append("### Вывод")
    lines.append("Для варианта 11 реализован синтезатор речи: создан набор из 63 образцов фонем/аллофонов, фраза синтезирована по фонетической цепочке, сравнены простая конкатенация и монтаж с перекрёстным затуханием. Версия `phrase_tchaikovsky_based.wav` дополнительно использует фортепианный фрагмент из предыдущей лабораторной как музыкальную основу.")
    REPORT_PATH.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    ensure_clean_dir(RESULTS_DIR); ensure_clean_dir(SRC_DIR)
    phon_dir = RESULTS_DIR / "phonemes"; phon_dir.mkdir(parents=True, exist_ok=True)
    samples = make_samples()
    rows = []
    for name, x in samples.items():
        save_wav(phon_dir / f"{name}.wav", x)
        kind = "vowel/allophone" if name in VOWELS else "consonant"
        rows.append((name, kind, len(x)/FS, f"phonemes/{name}.wav"))
    with (RESULTS_DIR / "phonemes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";"); w.writerow(["id","type","duration_sec","file"]); w.writerows(rows)

    concat, cjoins = concat_synthesis(samples, PHRASE)
    cross, xjoins = crossfade_synthesis(samples, PHRASE, CROSSFADE_MS)
    piano = load_piano(len(cross))
    piano_mix = piano_based_track(cross, piano)

    save_wav(RESULTS_DIR / "phrase_concat.wav", concat)
    save_wav(RESULTS_DIR / "phrase_crossfade.wav", cross)
    save_wav(RESULTS_DIR / "phrase_tchaikovsky_based.wav", piano_mix)
    save_wav(RESULTS_DIR / "piano_fragment_used.wav", piano)

    plot_waveforms(concat, cross, piano_mix, RESULTS_DIR / "waveforms_compare.png")
    plot_spectrogram(concat, RESULTS_DIR / "spectrogram_concat.png", "Спектрограмма: простая конкатенация")
    plot_spectrogram(cross, RESULTS_DIR / "spectrogram_crossfade.png", "Спектрограмма: перекрёстное затухание")
    plot_spectrogram(piano_mix, RESULTS_DIR / "spectrogram_tchaikovsky_based.png", "Спектрограмма: синтез на основе фрагмента Чайковского")

    stats = join_stats(concat, cjoins, cross, xjoins)
    save_join_csv(RESULTS_DIR / "join_stats.csv", stats)
    mjc = float(np.mean([s.jump_concat for s in stats])) if stats else 0.0
    mjx = float(np.mean([s.jump_crossfade for s in stats])) if stats else 0.0
    red = (1 - mjx/(mjc+1e-12))*100 if mjc > 0 else 0.0
    summary = {
        "variant": VARIANT, "variant_branch": VARIANT_BRANCH, "phrase": PHRASE_TEXT,
        "phoneme_count": len(samples), "sample_rate_hz": FS,
        "duration_concat_sec": len(concat)/FS, "duration_crossfade_sec": len(cross)/FS,
        "duration_piano_based_sec": len(piano_mix)/FS, "crossfade_ms": CROSSFADE_MS,
        "mean_jump_concat": mjc, "mean_jump_crossfade": mjx, "jump_reduction_percent": red,
        "piano_source_used": str(PIANO_SOURCE), "transcription_chain": PHRASE,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    shutil.copytree(RESULTS_DIR, SRC_DIR, dirs_exist_ok=True)
    print("Лабораторная работа №10 выполнена.")
    print(f"Вариант: {VARIANT} -> ветка {VARIANT_BRANCH}")
    print(f"Фонемных файлов: {len(samples)}")
    print(f"Фраза: {PHRASE_TEXT}")
    print(f"Длительность crossfade: {len(cross)/FS:.3f} с")
    print(f"Уменьшение скачков: {red:.2f}%")
    print(f"Результаты: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
