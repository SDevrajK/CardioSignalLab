"""EEMD (Ensemble Empirical Mode Decomposition) artifact removal.

Decomposes signal into Intrinsic Mode Functions (IMFs), classifies each by
frequency and energy content, excludes artifact IMFs, and reconstructs.

Ported from Shimmer_Testing/lib/preprocessing/emd_denoising.py.

References:
    Wu & Huang (2009): EEMD algorithm, noise_width=0.2 standard
    Motin et al. (2018): PPG artifact removal with EEMD
"""
from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy.signal import welch

from cardio_signal_lab.processing.pipeline import register_operation


def eemd_decompose(
    signal: np.ndarray,
    ensemble_size: int = 500,
    noise_width: float = 0.2,
    max_imf: int = -1,
    random_seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose signal into IMFs using EEMD.

    Args:
        signal: Input signal (1D array, >= 100 samples)
        ensemble_size: Number of ensemble trials (higher = better separation,
                       slower). 500 is standard balance of quality/speed.
        noise_width: Added noise amplitude as fraction of signal std.
                     0.2 is Wu & Huang (2009) standard.
        max_imf: Maximum number of IMFs (-1 = auto-determine)
        random_seed: Seed for reproducibility (None = random)

    Returns:
        Tuple of (IMFs array [n_imfs x n_samples], residue array [n_samples])

    Raises:
        ValueError: If signal too short (< 100 samples)
    """
    from PyEMD import EEMD

    if len(signal) < 100:
        raise ValueError(f"Signal too short for EEMD: {len(signal)} samples (need >= 100)")

    eemd = EEMD()
    eemd.noise_seed(random_seed)

    logger.info(
        f"Starting EEMD decomposition: {len(signal)} samples, "
        f"ensemble_size={ensemble_size}, noise_width={noise_width}"
    )

    all_imfs = eemd.eemd(signal, max_imf=max_imf)

    # Last row is the residue
    imfs = all_imfs[:-1]
    residue = all_imfs[-1]

    logger.info(f"EEMD decomposition complete: {imfs.shape[0]} IMFs extracted")
    return imfs, residue


def analyze_imf_characteristics(
    imfs: np.ndarray,
    sampling_rate: float = 64.0,
) -> list[dict[str, Any]]:
    """Compute frequency and energy metrics for each IMF.

    Uses Welch's method for power spectral density estimation and
    zero-crossing analysis for period estimation.

    Args:
        imfs: IMF array [n_imfs x n_samples]
        sampling_rate: Signal sampling rate in Hz

    Returns:
        List of dicts with keys: imf_index, peak_freq, energy, energy_pct,
        mean_period
    """
    total_energy = np.sum(imfs ** 2)
    characteristics = []

    for i in range(imfs.shape[0]):
        imf = imfs[i]
        energy = float(np.sum(imf ** 2))
        energy_pct = 100.0 * energy / total_energy if total_energy > 0 else 0.0

        # Welch PSD for peak frequency
        nperseg = min(256, len(imf) // 4)
        if nperseg < 4:
            nperseg = len(imf)
        f, psd = welch(imf, fs=sampling_rate, nperseg=nperseg)
        peak_freq = float(f[np.argmax(psd)])

        # Zero-crossing period estimation
        zero_crossings = np.where(np.diff(np.sign(imf)))[0]
        if len(zero_crossings) > 1:
            mean_period = float(np.mean(np.diff(zero_crossings)) / sampling_rate)
        else:
            mean_period = 0.0

        characteristics.append({
            "imf_index": i,
            "peak_freq": peak_freq,
            "energy": energy,
            "energy_pct": energy_pct,
            "mean_period": mean_period,
        })

    return characteristics


def auto_select_artifact_imfs(
    characteristics: list[dict[str, Any]],
) -> list[int]:
    """Auto-select IMFs to exclude based on frequency and energy.

    Classification rules (from Shimmer_Testing):
    - High-frequency noise (>= 15 Hz): always exclude
    - Spike/transient (energy < 0.5% and freq > 10 Hz): exclude
    - Artifact range (4-8 Hz, energy < 15%): exclude
    - Cardiac content (0.67-3.0 Hz): keep
    - Respiratory/drift: keep

    Args:
        characteristics: Output from analyze_imf_characteristics()

    Returns:
        List of 0-based IMF indices to exclude
    """
    exclude = []

    for char in characteristics:
        idx = char["imf_index"]
        freq = char["peak_freq"]
        energy_pct = char["energy_pct"]

        # Priority 1: Very low energy, high frequency spikes
        if energy_pct < 0.5 and freq > 10:
            exclude.append(idx)
            logger.debug(f"IMF {idx}: EXCLUDE (spike/transient, {freq:.1f} Hz, {energy_pct:.1f}%)")

        # Priority 2: High-frequency noise
        elif freq >= 15:
            exclude.append(idx)
            logger.debug(f"IMF {idx}: EXCLUDE (high-freq noise, {freq:.1f} Hz)")

        # Priority 3: Artifact range with low energy
        elif 4.0 <= freq <= 8.0 and energy_pct < 15:
            exclude.append(idx)
            logger.debug(f"IMF {idx}: EXCLUDE (artifact range, {freq:.1f} Hz, {energy_pct:.1f}%)")

        else:
            logger.debug(f"IMF {idx}: KEEP ({freq:.1f} Hz, {energy_pct:.1f}%)")

    logger.info(f"Auto-selected {len(exclude)} IMFs to exclude: {exclude}")
    return exclude


def reconstruct_from_imfs(
    imfs: np.ndarray,
    residue: np.ndarray,
    exclude_imfs: list[int] | None = None,
    include_residue: bool = True,
) -> np.ndarray:
    """Reconstruct signal from selected IMFs.

    Sums all IMFs except those in exclude_imfs, optionally adding the residue
    (baseline trend component).

    Args:
        imfs: IMF array [n_imfs x n_samples]
        residue: Residue signal [n_samples]
        exclude_imfs: 0-based indices of IMFs to exclude
        include_residue: Whether to include residue (baseline) in reconstruction

    Returns:
        Reconstructed signal
    """
    exclude_set = set(exclude_imfs or [])
    selected = [imfs[i] for i in range(imfs.shape[0]) if i not in exclude_set]

    if not selected:
        reconstructed = np.zeros_like(residue)
    else:
        reconstructed = np.sum(selected, axis=0)

    if include_residue:
        reconstructed += residue

    return reconstructed


def eemd_artifact_removal(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    ensemble_size: int = 500,
    noise_width: float = 0.2,
    random_seed: int | None = 42,
    exclude_imfs: list[int] | None = None,
) -> np.ndarray:
    """Complete EEMD artifact removal pipeline.

    Decomposes signal, auto-classifies IMFs (unless exclude_imfs provided),
    and reconstructs without artifact components.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        ensemble_size: EEMD ensemble size
        noise_width: EEMD noise amplitude ratio
        random_seed: Random seed for reproducibility
        exclude_imfs: Explicit IMF indices to exclude (None = auto-select)

    Returns:
        Denoised signal
    """
    imfs, residue = eemd_decompose(
        samples,
        ensemble_size=ensemble_size,
        noise_width=noise_width,
        random_seed=random_seed,
    )

    characteristics = analyze_imf_characteristics(imfs, sampling_rate)

    if exclude_imfs is None:
        exclude_imfs = auto_select_artifact_imfs(characteristics)

    reconstructed = reconstruct_from_imfs(imfs, residue, exclude_imfs=exclude_imfs)

    # Log quality metrics
    correlation = float(np.corrcoef(samples, reconstructed)[0, 1])
    rmse = float(np.sqrt(np.mean((samples - reconstructed) ** 2)))
    logger.info(
        f"EEMD artifact removal complete: excluded {len(exclude_imfs)} IMFs, "
        f"correlation={correlation:.4f}, RMSE={rmse:.4f}"
    )

    return reconstructed


# Pipeline wrapper
def _pipeline_eemd(samples, sampling_rate, **params):
    return eemd_artifact_removal(samples, sampling_rate, **params)


register_operation("eemd_artifact_removal", _pipeline_eemd)
