# Error Handling Patterns from EKG_Peak_Corrector v2

**Date**: 2026-02-15
**Source**: EKG_Peak_Corrector v2 codebase analysis
**Purpose**: Identify error handling patterns to adopt for CardioSignalLab MVP

---

## 1. FILE LOADING ERROR HANDLING

**Pattern - Layered Input Validation** (xdf_loader.py:50-56):
```python
# Type validation first
if not isinstance(filepath, str):
    raise TypeError(f"filepath must be a string, got {type(filepath).__name__}")

# File existence check
if not os.path.exists(filepath):
    raise FileNotFoundError(f"XDF file not found: {filepath}")

# File format validation
if not filepath.endswith('.xdf'):
    raise ValueError(f"Invalid file extension, expected .xdf: {filepath}")
```

**Key Principles**:
- Validate early and specifically (type → existence → format)
- Provide specific error messages with actual/expected values
- Use appropriate exception types (TypeError, FileNotFoundError, ValueError)

**Fallback Pattern** (xdf_loader.py:112-127):
```python
try:
    # Try selective loading first
    streams, header = pyxdf.load_xdf(filepath, select_streams=select_queries)
    if not streams:
        # Fallback to loading all streams
        streams, header = pyxdf.load_xdf(filepath)
except Exception as e:
    # Log warning and retry with simpler approach
    logger.warning(f"Selective loading failed ({e}), falling back...")
    streams, header = pyxdf.load_xdf(filepath)
```

---

## 2. SIGNAL DATA VALIDATION

**Pattern - Dimension and Range Validation** (ecg_processor.py:30-36):
```python
def find_best_continuous_segment(self, signal_data, data_mask, sampling_rate):
    # Array type checks
    if not isinstance(signal_data, np.ndarray) or not isinstance(data_mask, np.ndarray):
        raise TypeError("signal_data and data_mask must be numpy arrays")

    # Dimension alignment
    if len(signal_data) != len(data_mask):
        raise ValueError(f"signal_data and data_mask must have same length")

    # Range validation
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive number")
```

**Key Checks**:
- Data type validation (numpy arrays, floats)
- Shape/dimension alignment
- Range validation (sampling_rate > 0)

---

## 3. BAD SEGMENT DETECTION

**Pattern - Gap Classification by Duration** (gap_detection.py:37-78):
```python
MICRO_GAP_THRESHOLD = 0.05        # 50ms
SMALL_GAP_THRESHOLD = 0.5         # 500ms
MEDIUM_GAP_THRESHOLD = 4.0        # 4s

def classify_gap(self, gap_duration, sampling_rate):
    if gap_duration < self.MICRO_GAP_THRESHOLD:
        return "intra_beat"        # Within QRS/T-wave
    elif gap_duration < self.SMALL_GAP_THRESHOLD:
        return "partial_beat"      # Part of beat missing
    elif gap_duration < self.MEDIUM_GAP_THRESHOLD:
        return "few_beats"         # Few cycles missing
    else:
        return "many_beats"        # Many cycles missing
```

**Vectorized Gap Detection** (gap_detection.py:107-110):
```python
time_diffs = np.diff(irregular_timestamps)
gap_threshold = expected_interval * 1.5  # Configurable threshold
gap_indices = np.where(time_diffs > gap_threshold)[0]  # Vectorized
```

---

## 4. QUALITY CHECKS WITH GRACEFUL FALLBACK

**Pattern - Try Quality Calculation with Fallback** (xdf_loader.py:751-753):
```python
try:
    ppg_quality = nk.ppg_quality(ppg_signal, sampling_rate=self.ppg_target_rate)
    logger.info(f"Calculated PPG quality: mean={np.mean(ppg_quality):.3f}")
except Exception as e:
    logger.warning(f"Failed to calculate PPG quality: {e}")
    ppg_quality = None  # Continue without quality scores
```

**Application**:
- Attempt quality scoring, but don't block signal loading if it fails
- Mark signals as "unscored" rather than failing
- Log warnings for analysis but continue processing

---

## 5. VALIDATION WITH MEANINGFUL MESSAGES

**Pattern - Clear Validation with Context** (data_models.py:199-222):
```python
def set_ecg_quality(self, quality):
    if len(quality) != self.ecg_sample_count:
        raise ValueError(
            f"Quality array length ({len(quality)}) must match "
            f"ECG data length ({self.ecg_sample_count})"
        )
```

**Best Practices**:
- Always report: actual_value vs. expected_value
- Include context about what the signal is (channel name, data type)
- Suggest remediation (e.g., "need ≥5 seconds for reliable analysis")

---

## 6. STAGE-BY-STAGE FAILURE HANDLING

**Pattern - Process Pipeline with Stage Tracking** (batch_processor.py:96-175):
```python
def process_file(self, mode='multi-channel'):
    try:
        # Stage 1: Load
        if not self.load_file(Path(self.xdf_file_path)):
            return False

        # Stage 2: Select channels
        if not self.select_preferred_channels():
            return False

        # ... continue stages ...

    except Exception as e:
        print(f"CRITICAL ERROR in processing pipeline")
        print(f"Error: {str(e)}")
        if self.verbose:
            print(f"Traceback: {traceback.format_exc()}")
        return False

# Stage tracking
self.stage_status = {
    'load_file': False,
    'select_channels': False,
    'detect_gaps': False,
}
```

**Application for CardioSignalLab**:
- Track processing stages (File loaded → Validated → Filtered → Peaks detected)
- Report stage where failure occurred
- Store full tracebacks in logs, but show concise messages in GUI
- Allow resume from last successful stage

---

## 7. COMMON FAILURE MODES

| Failure Mode | Detection | Handling |
|---|---|---|
| **Missing ECG channels** | Check channel list is non-empty | Fail with clear message listing available channels |
| **Misaligned timestamps** | Compare ECG/PPG array lengths | Crop to shorter duration automatically |
| **NaN values in resampled data** | Check `np.isnan()` on signals | Interpolate linearly or flag as unreliable |
| **No continuous segments >30s** | Gap detection finds only small gaps | Suggest reducing MIN_SEGMENT_DURATION |
| **Insufficient peaks for analysis** | Count detected peaks < 3 | Warn "Peak detection unreliable" |
| **Invalid file format** | pyxdf.load_xdf() exception | Fallback to stream selection retry |

---

## ERROR HANDLING ARCHITECTURE FOR CARDIOSIGNALLAB

### 1. Layered Validation (Bottom-Up)
```
File I/O Layer:
  - File exists / readable ✓
  - Format correct (XDF/CSV valid)

Data Parsing Layer:
  - Streams contain ECG/PPG
  - Timestamps are strictly increasing

Data Validation Layer:
  - Dimensions match (ECG == PPG)
  - Value ranges reasonable
  - No excessive NaN/gaps

Processing Layer:
  - Sufficient peak detection
  - Quality scores available
```

### 2. Error Classification
```
CRITICAL (stop processing):
  - File not found
  - No ECG data in file
  - Corrupted data structure

WARNING (log but continue):
  - PPG quality calculation failed
  - Timestamp discontinuities detected
  - Peak detection unreliable (< 3 peaks)

INFO (log for debugging):
  - NaN values interpolated
  - Channels resampled
  - Stage completion time
```

### 3. User-Facing Error Messages
```
BAD:   "ValueError: Dimension mismatch"
GOOD:  "ECG has 8192 samples but PPG has 4096.
        Cropping PPG to match ECG."

BAD:   "peaks < 3 failed"
GOOD:  "Warning: Only 2 peaks detected. Peak detection
        may be unreliable. Suggest manual review."

BAD:   "pyxdf load failed: (some traceback)"
GOOD:  "File format unrecognized. Trying fallback
        CSV loader... Success!"
```

### 4. Exception Handling Style
```python
# From v2 - follow this pattern:
try:
    result = operation()
except SpecificException as e:
    logger.warning(f"Operation failed: {e}")
    result = fallback_operation()
except Exception as e:  # Only as final catch-all
    logger.error(f"Unexpected error: {e}")
    raise  # Re-raise critical errors
```

---

## IMPLEMENTATION CHECKLIST FOR CARDIOSIGNALLAB

### File Loaders (Task 7.2)
- [ ] Add layered validation to XdfLoader: type → exists → format
- [ ] Add validation to CsvLoader: column format, header presence
- [ ] Add fallback CSV loading if JSON fails
- [ ] Check sampling rates are positive and reasonable (32-512 Hz)
- [ ] Validate timestamps are strictly increasing
- [ ] Detect empty signals (all zeros, constant values)
- [ ] Log warnings via loguru for skipped streams

### Processing Pipeline (Task 7.3)
- [ ] Wrap filter operations in try/except
- [ ] Detect signal dropouts (gaps in timestamps)
- [ ] Mark problematic segments instead of failing
- [ ] Continue processing clean segments
- [ ] Emit processing_finished even on partial failure

### GUI Log Panel (Task 7.4)
- [ ] Add QTextEdit dock widget
- [ ] Add custom loguru sink for real-time log display
- [ ] Toggle visibility via View menu
- [ ] Write to both panel and log file

---

## SUMMARY

1. **Validate early**: File → Format → Data structure → Values
2. **Be specific**: Use ValueError/TypeError/FileNotFoundError, not generic Exception
3. **Log context**: Include actual vs. expected values in error messages
4. **Fail gracefully**: Fallback loaders, skip optional features, mark as "unscored"
5. **Track progress**: Record which processing stages succeeded/failed
6. **Inform user**: Clear messages for GUI, detailed logs for debugging
7. **Test boundaries**: Check array lengths, value ranges, timestamp monotonicity

These patterns are battle-tested against real physiological data with dropouts, channel misalignments, and format variations from multiple hardware vendors.
