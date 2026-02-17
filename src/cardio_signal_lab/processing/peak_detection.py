"""NeuroKit2 wrappers for ECG, PPG, and EDA peak detection.

Uses step-by-step NeuroKit2 functions (clean -> peaks) for better control
than the all-in-one process functions. Returns peak indices compatible with
PeakData model.

Ported from Hyperacousie_TCC and Acute_Tinnitus_PPG NeuroKit2 patterns.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from cardio_signal_lab.processing.pipeline import register_operation


def detect_ecg_peaks(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    clean_method: str = "neurokit",
    peak_method: str = "neurokit",
) -> np.ndarray:
    """Detect R-peaks in ECG signal using NeuroKit2.

    Pipeline: clean -> detect peaks -> return indices.
    Default methods tested in Hyperacousie_TCC project.

    Args:
        samples: ECG signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        clean_method: NeuroKit2 cleaning method
            ('neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002')
        peak_method: Peak detection method
            ('neurokit', 'pantompkins1985', 'hamilton2002', 'martinez2004')

    Returns:
        Array of peak indices (sample positions)
    """
    import neurokit2 as nk

    try:
        # Step 1: Clean ECG signal
        ecg_cleaned = nk.ecg_clean(samples, sampling_rate=int(sampling_rate), method=clean_method)

        # Step 2: Detect R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(sampling_rate), method=peak_method)

        peak_indices = rpeaks.get("ECG_R_Peaks", np.array([], dtype=int))
        if peak_indices is None:
            peak_indices = np.array([], dtype=int)
        peak_indices = np.asarray(peak_indices, dtype=int)

        logger.info(
            f"ECG peak detection: {len(peak_indices)} R-peaks found "
            f"(clean={clean_method}, detect={peak_method})"
        )

        if len(peak_indices) < 3:
            logger.warning(
                f"Only {len(peak_indices)} peaks detected - may indicate signal quality issues "
                f"or inappropriate parameters"
            )

        return peak_indices

    except Exception as e:
        logger.error(f"ECG peak detection failed: {e}")
        logger.warning("Returning empty peak array - check signal quality and sampling rate")
        return np.array([], dtype=int)


def detect_ppg_peaks(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    clean_method: str = "elgendi",
    peak_method: str = "elgendi",
) -> np.ndarray:
    """Detect systolic peaks in PPG signal using NeuroKit2.

    Pipeline: clean -> find peaks -> return indices.

    Args:
        samples: PPG signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        clean_method: NeuroKit2 cleaning method ('elgendi')
        peak_method: Peak detection method ('elgendi')

    Returns:
        Array of peak indices (sample positions)
    """
    import neurokit2 as nk

    try:
        # Step 1: Clean PPG signal
        ppg_cleaned = nk.ppg_clean(samples, sampling_rate=int(sampling_rate), method=clean_method)

        # Step 2: Detect peaks
        info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=int(sampling_rate), method=peak_method)

        peak_indices = info.get("PPG_Peaks", np.array([], dtype=int))
        if peak_indices is None:
            peak_indices = np.array([], dtype=int)
        peak_indices = np.asarray(peak_indices, dtype=int)

        logger.info(
            f"PPG peak detection: {len(peak_indices)} peaks found "
            f"(clean={clean_method}, detect={peak_method})"
        )

        if len(peak_indices) < 3:
            logger.warning(
                f"Only {len(peak_indices)} peaks detected - may indicate signal quality issues "
                f"or inappropriate parameters"
            )

        return peak_indices

    except Exception as e:
        logger.error(f"PPG peak detection failed: {e}")
        logger.warning("Returning empty peak array - check signal quality and sampling rate")
        return np.array([], dtype=int)


def detect_eda_features(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    method: str = "neurokit",
    decompose_method: str = "highpass",
) -> np.ndarray:
    """Detect SCR (Skin Conductance Response) peaks in EDA signal.

    Uses nk.eda_process() for full processing, then extracts SCR peak indices.

    Args:
        samples: EDA signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        method: NeuroKit2 processing method
        decompose_method: Phasic decomposition method ('highpass')

    Returns:
        Array of SCR peak indices (sample positions)
    """
    import neurokit2 as nk

    try:
        # Process EDA (clean + decompose + detect SCR peaks)
        signals, info = nk.eda_process(samples, sampling_rate=int(sampling_rate), method=method)

        peak_indices = info.get("SCR_Peaks", np.array([], dtype=int))
        if peak_indices is None:
            peak_indices = np.array([], dtype=int)
        peak_indices = np.asarray(peak_indices, dtype=int)

        # Filter out NaN values that NeuroKit2 sometimes returns
        peak_indices = peak_indices[~np.isnan(peak_indices.astype(float))].astype(int)

        logger.info(f"EDA feature detection: {len(peak_indices)} SCR peaks found")

        if len(peak_indices) == 0:
            logger.warning(
                "No SCR peaks detected - EDA signal may lack phasic responses "
                "or may be too short/noisy"
            )

        return peak_indices

    except Exception as e:
        logger.error(f"EDA feature detection failed: {e}")
        logger.warning("Returning empty peak array - check signal quality and sampling rate")
        return np.array([], dtype=int)


# Pipeline wrappers that return the processed signal unchanged but store peak
# indices as a side effect. Since peak detection doesn't modify the signal
# itself, we use a different approach: these are called directly from the GUI,
# not through the pipeline. But we register them for recording in processing
# history.

def _pipeline_detect_ecg_peaks(samples, sampling_rate, **params):
    """Pipeline wrapper - detects peaks but returns signal unchanged.

    Peak indices are stored via AppSignals.peaks_updated by the GUI layer.
    """
    detect_ecg_peaks(samples, sampling_rate, **params)
    return samples


def _pipeline_detect_ppg_peaks(samples, sampling_rate, **params):
    detect_ppg_peaks(samples, sampling_rate, **params)
    return samples


def _pipeline_detect_eda_features(samples, sampling_rate, **params):
    detect_eda_features(samples, sampling_rate, **params)
    return samples


register_operation("detect_ecg_peaks", _pipeline_detect_ecg_peaks)
register_operation("detect_ppg_peaks", _pipeline_detect_ppg_peaks)
register_operation("detect_eda_features", _pipeline_detect_eda_features)


def _pipeline_ecg_clean(samples, sampling_rate, **params):
    """Pipeline wrapper for nk.ecg_clean()."""
    import neurokit2 as nk
    return nk.ecg_clean(samples, sampling_rate=int(sampling_rate))


def _pipeline_ppg_clean(samples, sampling_rate, **params):
    """Pipeline wrapper for nk.ppg_clean()."""
    import neurokit2 as nk
    return nk.ppg_clean(samples, sampling_rate=int(sampling_rate))


def _pipeline_eda_clean(samples, sampling_rate, **params):
    """Pipeline wrapper for nk.eda_clean()."""
    import neurokit2 as nk
    return nk.eda_clean(samples, sampling_rate=int(sampling_rate))


def _pipeline_eda_decompose(samples, sampling_rate, component="phasic", method="highpass", **params):
    """Pipeline wrapper for nk.eda_process() - returns the selected component."""
    import neurokit2 as nk
    signals_df, _ = nk.eda_process(samples, sampling_rate=int(sampling_rate), method=method)
    col = "EDA_Phasic" if component == "phasic" else "EDA_Tonic"
    return signals_df[col].to_numpy()


register_operation("ecg_clean", _pipeline_ecg_clean)
register_operation("ppg_clean", _pipeline_ppg_clean)
register_operation("eda_clean", _pipeline_eda_clean)
register_operation("eda_decompose", _pipeline_eda_decompose)
