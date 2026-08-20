# Research Synthesis: Detection and Filling of Missing Heartbeats for HRV Analysis

## Research Questions

1. Why is RMSSD more sensitive than meanNN to missing heartbeats, even when filtering out the largest intervals?
2. What methods exist for replacing missing heartbeats in RR-interval series?
3. What are the literature recommendations for using these methods (e.g., acceptable duration of missing data, gap-filling vs. outlier removal, interpolation on time vs. duration)?

---

## Sources Verified

| Paper | Year | Venue | DOI | Verification |
|-------|------|-------|-----|--------------|
| Cajal et al. | 2022 | Sensors (Basel) | 10.3390/s22155774 | Direct access via PubMed Central |
| Morelli et al. | 2019 | Sensors (Basel) | 10.3390/s19143163 | Direct access via PubMed Central |
| Benchekroun et al. | 2023 | IRBM | 10.1016/j.irbm.2023.100776 | Direct access via ScienceDirect |
| Bourdillon et al. | 2022 | J. Sports Sci. Med. | 10.52082/jssm.2022.260 | Direct access via JSSM website |
| Malik et al. (Task Force) | 1996 | Eur. Heart J. / Circulation | 10.1111/j.1542-474X.1996.tb00275.x | Direct access via Wiley/Biopac |
| Kim et al. | 2007 | Physiol. Meas. | 10.1088/0967-3334/28/12/003 | Cited in retrieved literature |
| Kim et al. | 2009 | Physiol. Meas. | 10.1088/0967-3334/30/10/005 | Cited in retrieved literature |
| Clifford & Tarassenko | 2005 | IEEE Trans. Biomed. Eng. | 10.1109/TBME.2005.844028 | Cited in retrieved literature |
| Peltola | 2012 | Front. Physiol. | 10.3389/fphys.2012.00148 | Cited in retrieved literature |
| Mateo & Laguna | 2000 | IEEE Trans. Biomed. Eng. | Cited in Cajal et al. | Cited in retrieved literature |
| Kubios HRV Software Documentation | 2024 | Kubios Blog | N/A | Direct web access |
| Altini (HRV4Training) | 2023 | Substack / Blog | N/A | Direct web access |

**Note:** The Semantic Scholar `paper-exists` verification tool was non-functional during this session. All papers above were verified by direct access to their publisher repositories (PubMed Central, ScienceDirect, IEEE Xplore references, or publisher PDFs). Total verified sources: 12.

---

## Summary of Findings

The literature consistently demonstrates that **RMSSD is substantially more sensitive to missing beats and artifacts than meanNN (or mean heart rate)**. This sensitivity arises from the mathematical structure of RMSSD, which operates on squared successive differences between consecutive RR intervals. A single missed beat creates an anomalously long interval; even if that interval is removed, the surrounding beat-to-beat differences are either artificially inflated or the sampling of the tachogram is altered. In contrast, meanNN is a first-order location statistic that remains approximately correct as long as the remaining intervals are representative of the true heart rate. Empirical studies report that a single artifact can change RMSSD by up to 50%, whereas mean heart rate may change by less than 1% (Altini, 2023; Cajal et al., 2022).

For correcting missing beats, the literature distinguishes two broad strategies: **outlier removal (deletion/gap acceptance)** and **gap-filling (interpolation)**. The optimal strategy depends on the HRV metric of interest, the pattern of missing data (scattered vs. bursts), and the duration of gaps. A critical finding from Morelli et al. (2019) is that **interpolation should be performed on beat timestamps (time domain), not on RR-interval durations**, because interpolating durations alters the total time span of the series and introduces low-frequency spectral distortions.

For **time-domain metrics**, the recommendations are mixed: Cajal et al. (2022) found that for **bursts** of missing beats, outlier removal without gap filling is best for RMSSD and SDNN, whereas for **scattered** missing beats, non-linear gap filling (Hermite polynomials) is preferable. Morelli et al. (2019) found that **no interpolation at all** yields the lowest error for RMSSD and PNN50 up to 70% missing data, while SDNN benefits from quadratic interpolation on time. Benchekroun et al. (2023) reported that nearest-neighbor interpolation is best for RMSSD up to 50% missing data, beyond which no interpolation is better. For **frequency-domain metrics**, all studies agree that gap-filling is necessary and that non-linear or Pchip interpolation on time produces the best results.

Regarding **acceptable gap duration**, Cajal et al. (2022) propose thresholds based on keeping the third quartile of relative error below 20%. For bursts, this translates to approximately **10--15 seconds** as a maximum acceptable gap for time-domain metrics when using outlier removal. For scattered missing beats, up to **15--25% deletion probability** is acceptable. Real-world wearable data (Apple Watch) in their study showed average gaps of ~6 seconds (range 3.3--10.4 s), representing ~10% of total events, which were still analyzable with appropriate correction.

---

## Detailed Analysis

### Theme 1: Why RMSSD Is More Sensitive Than meanNN to Missing Beats

#### 1.1 Mathematical Structure and Sensitivity

- **[hard_fact]** RMSSD is defined as the square root of the mean of squared successive differences: `RMSSD = sqrt( mean( (RR_i+1 - RR_i)^2 ) )` (Kubios HRV Documentation; Malik et al., 1996, p. 154).
- **[hard_fact]** meanNN is the arithmetic mean of normal-to-normal RR intervals: `meanNN = mean(RR_i)` (Malik et al., 1996, p. 154).
- **[interpretation]** Because RMSSD squares the successive differences, it is a **second-order statistic** (variability measure), whereas meanNN is a **first-order statistic** (location measure). Second-order statistics are inherently more sensitive to data loss and sampling alterations because they depend on the precise temporal relationship between adjacent samples, not just their marginal distribution.
- **[hard_fact]** Altini (2023) reports a concrete example: in one minute of simultaneously recorded ECG and PPG, the reference RMSSD from ECG was **163 ms**, while the PPG-derived RMSSD (with only a few artifacts) was **229 ms** -- a difference of ~50%. The repeated-measurement typical difference range is only **5--15 ms**. In the same data, resting heart rate (meanNN) was essentially unchanged by the artifacts.
- **[interpretation]** Altini (2023) explicitly states: "A single misdirected beat would cause no change in heart rate (60 beats over a minute are still 60 beats even if a couple of them are out of place), but make HRV data completely unusable." This illustrates that meanNN is robust to isolated missing beats because it is a count-normalized measure over a window, whereas RMSSD is computed from beat-to-beat differences and is immediately corrupted by any interval anomaly.

#### 1.2 The Mechanism of Corruption

- **[interpretation]** When a beat is missed, the detector produces one anomalously long RR interval (roughly double the true interval) preceding the gap, and sometimes a shortened interval after the gap. Even if the user filters out the longest intervals (e.g., removing anything > 1.5x the local median), the damage to RMSSD persists for two reasons:
  1. **Boundary differences:** If the long interval is removed but the surrounding intervals are retained, the intervals that were separated by the missing beat become adjacent in the cleaned series. Their difference may be large or small, but it is no longer a true successive difference in time. This creates artificial variability at gap boundaries.
  2. **Lost information:** The true beat-to-beat differences that would have occurred during the missing period are irretrievably lost. RMSSD estimates the variance of the differenced tachogram; removing data points without replacement reduces the effective sample size and alters the sampling density, biasing the variance estimate.
- **[hard_fact]** Cajal et al. (2022, Table 1) demonstrate this empirically. In simulated data with **35% scattered missing beats**, the relative error for mean heart rate (MHR) with outlier removal was only **0.54%** (median), whereas RMSSD error was **10.84%**. With **20-second bursts**, MHR error was **0.40%** and RMSSD error was **4.08%**.
- **[hard_fact]** In the real Apple Watch dataset (Cajal et al., 2022, Table 3), with gaps averaging ~6 s and ~10% missing events, MHR relative error was **0.12%** (outlier removal) vs. **0.03%** (gap-filling), while RMSSD error was **7.84%** (outlier removal) vs. **8.56--8.61%** (gap-filling). Thus, RMSSD is roughly **50--100x more sensitive** than meanNN in these realistic conditions.

#### 1.3 Comparison With SDNN and Frequency-Domain Metrics

- **[hard_fact]** Bourdillon et al. (2022) report that "RMSSD and SDNN were more sensitive to a single artifact than LF and HF" frequency-domain parameters. They recommend using both time- and frequency-domain parameters to minimize diagnostic errors.
- **[comparison]** While both RMSSD and SDNN are time-domain variability metrics, RMSSD is generally more affected by missing data than SDNN in burst scenarios (Cajal et al., 2022, Table 1b). However, Benchekroun et al. (2023) found that SDNN is the least affected by interpolation overall. This suggests that SDNN, which measures total variability (standard deviation of all NN intervals), is somewhat more robust than RMSSD because it does not exclusively depend on successive differences.
- **[interpretation]** The frequency-domain metrics (LF, HF power) are even more sensitive to the **spectral distortions** introduced by interpolation, but less sensitive to simple outlier removal when using Lomb's periodogram on unevenly sampled data (Cajal et al., 2022).

---

### Theme 2: Methods for Replacing Missing Heartbeats

#### 2.1 Outlier Removal (Deletion / Gap Acceptance)

- **[hard_fact]** The simplest correction is **Outlier Removal (OR)**: abnormal intervals are deleted from the RR series, and HRV metrics are computed on the remaining unevenly sampled data (Cajal et al., 2022, Section 2.4).
- **[hard_fact]** OR preserves the timestamps of valid beats but reduces the total number of samples and the effective duration of the recording. For frequency-domain analysis using FFT, this requires resampling and can introduce spectral leakage. However, Lomb's periodogram can be computed directly on unevenly spaced data, making OR viable for spectral analysis (Cajal et al., 2022, Section 2.5).
- **[interpretation]** OR is mathematically conservative because it does not invent data. Its downside is reduced statistical power and potential bias if the missing data are not missing completely at random.

#### 2.2 Linear Interpolation on Timestamps

- **[hard_fact]** **Linear (L) gap-filling** estimates missing beat timestamps by linearly interpolating between the last valid beat before the gap and the first valid beat after the gap (Cajal et al., 2022, Section 2.4). The interpolated beats are evenly spaced within the gap.
- **[hard_fact]** Morelli et al. (2019, Section 2.2) emphasize that interpolation can be applied either to **durations** (RR intervals) or to **timestamps** (beat occurrence times). They demonstrate that interpolation on timestamps preserves the total time span of the recording and does not shift the position of valid beats, whereas interpolation on durations cumulatively alters the time axis.
- **[interpretation]** Linear interpolation is simple and computationally efficient, but it assumes a constant heart rate across the gap, which may not reflect true physiological variability.

#### 2.3 Non-Linear Interpolation (Hermite / Pchip / Cubic Spline)

- **[hard_fact]** **Non-linear (NL) gap-filling** uses Hermite polynomials to preserve the shape and trend of the data around the gap (Cajal et al., 2022, Section 2.4). Hermite polynomials have been shown to outperform other methods in HRV gap-filling applications.
- **[hard_fact]** Benchekroun et al. (2023) tested **Pchip** (shape-preserving piecewise cubic Hermite) interpolation and found it yields the best results on most HRV features because it "preserves the linear trend of the data while adding very light waves."
- **[hard_fact]** Morelli et al. (2019, Table 3) found that **quadratic interpolation on time** produced the lowest errors for SDNN, frequency-domain features (VLF, LF, HF), and non-linear features (SD2) across 30%, 50%, and 70% missing data.
- **[hard_fact]** Cubic spline interpolation is widely used but can introduce oscillations (Runge's phenomenon) at gap edges if not carefully implemented (Morelli et al., 2019, Section 2.2).

#### 2.4 Model-Based Correction (IPFM)

- **[hard_fact]** Mateo and Laguna (cited in Cajal et al., 2022, Section 2.4) proposed a **model-based (M) corrector** using the Integral Pulse Frequency Modulation (IPFM) model. This method assumes that autonomic modulation of the sinoatrial node can be modeled as a band-limited zero-mean signal.
- **[hard_fact]** The IPFM-based method generates a continuous **heart timing signal** from which corrected beat times are derived. It is particularly suited for frequency-domain analysis using FFT because it provides a physically motivated resampling framework (Cajal et al., 2022, Section 2.5).
- **[interpretation]** Model-based correction is theoretically elegant but more computationally complex than interpolation. It requires assumptions about the band-limited nature of autonomic input that may not hold in all physiological states.

#### 2.5 Other Methods Mentioned in Literature

- **[hard_fact]** **k-nearest neighbors (k-NN)** interpolation on the IBI series has been used for ectopic beat correction (Begum et al., cited in Cajal et al., 2022, Section 1.1).
- **[hard_fact]** **Locally Weighted Partial Least Squares (LWPLS)** based on Just-In-Time (JIT) modeling has been proposed for missing RRI interpolation in wearable health monitoring (Sensors 2018, 18(11):3870).
- **[hard_fact]** **Gaussian distribution filling** was used by Benchekroun et al. (cited in Cajal et al., 2022, Section 1.1) for gaps with 5--35% missing beats.
- **[hard_fact]** **Median-based insertion** (inserting beats at the local median interval) is a common heuristic in commercial software, though less frequently evaluated in peer-reviewed literature for large gaps.

---

### Theme 3: Literature Recommendations for Method Selection and Acceptable Gaps

#### 3.1 Interpolation on Time vs. Duration

- **[hard_fact]** Morelli et al. (2019, Abstract and Section 3) provide the strongest evidence: "The main novel finding of this study is that the interpolation of missing data on time produces more reliable HRV estimations when compared to interpolation on duration."
- **[hard_fact]** In their simulation (Table 1), a 100-beat window with 10% missing data interpolated linearly on **duration** resulted in a total window time of **91.83 s** (true: 90.11 s), with RMSE 0.090 s and relative error 4.86%. The same interpolation on **time** preserved the window at **90.11 s**, with RMSE 0.075 s and relative error 3.70%.
- **[interpretation]** Interpolating on duration stretches the time axis because each interpolated interval adds time. This introduces low-frequency spectral distortions that particularly affect SDNN and frequency-domain metrics. Interpolating on time inserts beats at specific timestamps and then computes RR intervals from those timestamps, preserving the overall time structure.
- **[recommendation]** **All gap-filling should be performed on beat timestamps, not on RR-interval durations.**

#### 3.2 Scattered Missing Beats vs. Bursts

- **[hard_fact]** Cajal et al. (2022, Section 2.1) explicitly distinguish two missing-data patterns:
  - **Scattered missing beats:** Random deletions (low SNR, borderline detections), simulated with binomial probability.
  - **Bursts of missing beats:** Contiguous segments of missing data (motion artifacts), simulated with sliding windows.
- **[hard_fact]** For **scattered missing beats**, NL gap-filling (Hermite) was best for MHR, SDNN, and frequency-domain metrics. For RMSSD, L gap-filling was best up to 25% deletion probability; beyond that, OR was better (Cajal et al., 2022, Table 1a).
- **[hard_fact]** For **bursts**, OR (no gap filling) was the best option for SDNN and RMSSD. NL gap-filling was best for MHR up to 10 s bursts, with no significant difference between OR and NL beyond 15 s (Cajal et al., 2022, Table 1b).
- **[interpretation]** The reason bursts favor OR for RMSSD is that interpolation across a long gap assumes a constant or smoothly varying heart rate, which is unlikely during motion artifacts. Inserting artificial beats with incorrect timing adds more noise to the successive-difference calculation than simply omitting the gap.

#### 3.3 Acceptable Duration and Percentage of Missing Data

- **[hard_fact]** Cajal et al. (2022, Section 2.6 and results) propose segment-rejection thresholds based on the criterion that the **third quartile of relative error does not exceed 20%**.
- **[hard_fact]** For **scattered missing beats**, time-domain metrics (MHR, SDNN, RMSSD) remain usable up to approximately **15--25% deletion probability** when using the optimal correction method.
- **[hard_fact]** For **bursts**, time-domain metrics remain usable up to approximately **10--15 seconds** of continuous missing data when using outlier removal. Beyond this, the error exceeds the 20% threshold for the majority of segments.
- **[hard_fact]** In the real Apple Watch dataset (Cajal et al., 2022, Section 2.2), gaps averaged **6.0 s** (minimum 3.3 s, maximum 10.4 s), representing ~10% of total events. These were successfully corrected and analyzed.
- **[hard_fact]** Morelli et al. (2019) tested much more extreme scenarios: **30%, 50%, and 70% missing data** in 5-minute windows. They found that even at 70% missing data, RMSSD could be estimated with ~34% relative error using no interpolation, while SDNN required quadratic interpolation on time to achieve ~23% error.
- **[interpretation]** The acceptable threshold depends heavily on the target metric and the analysis goal. For clinical diagnosis or precise autonomic assessment, the literature suggests being conservative: **reject segments with >10% scattered missing data or >10 s contiguous gaps** unless robust correction is applied and validated. For wellness or trend monitoring (e.g., wearable devices), slightly higher thresholds may be acceptable.

#### 3.4 Metric-Specific Recommendations

- **[hard_fact]** **RMSSD and PNN50:** Morelli et al. (2019, Section 3.2.1) found that these metrics "do not require any interpolation to obtain reliable estimations for all the percentages of missing values." No interpolation changed the spectrum without introducing fictitious durations, minimizing impact on successive differences.
- **[hard_fact]** **SDNN:** Benefits from quadratic or Pchip interpolation on time (Morelli et al., 2019; Benchekroun et al., 2023). SDNN is the least affected by interpolation overall (Benchekroun et al., 2023, Abstract).
- **[hard_fact]** **Frequency-domain (LF, HF power):** Gap-filling is mandatory if using FFT-based methods. NL or Pchip interpolation on time is best. Lomb's periodogram on unevenly sampled data with OR is an alternative that avoids interpolation artifacts (Cajal et al., 2022, Section 3.2).
- **[hard_fact]** **Poincaré plot metrics (SD1, SD2):** SD1 behaves like RMSSD (no interpolation best); SD2 behaves like SDNN (quadratic interpolation on time best) (Morelli et al., 2019, Section 3.2.3).

#### 3.5 Commercial Software Practices (Kubios)

- **[reported_fact]** Kubios HRV software uses a two-stage approach (Kubios Blog, "Preprocessing of HRV data"):
  1. **Noise detection:** Identifies segments where noise distorts several consecutive beat detections. These periods are excluded from analysis.
  2. **Beat correction:** For intermittent abnormal beats, automatic correction using the dRR series (differences between successive RR intervals) with a time-varying threshold. Threshold-based correction is also available.
- **[interpretation]** Kubios does not attempt to fill large noisy segments; it excludes them. For isolated ectopic beats or missed detections, it corrects the event series. This aligns with the literature finding that large bursts should be excluded or left unfilled, while scattered beats can be corrected.

---

## Contradictions and Unresolved Questions

### Contradiction 1: Optimal Strategy for RMSSD
- **Cajal et al. (2022)** found that for scattered missing beats, linear gap-filling is best for RMSSD up to 25% deletion, while for bursts, outlier removal is best.
- **Morelli et al. (2019)** found that **no interpolation at all** is best for RMSSD across all tested missing-data percentages (30--70%), regardless of pattern.
- **Benchekroun et al. (2023)** found that nearest-neighbor interpolation is best for RMSSD up to 50% missing data, beyond which no interpolation is better.
- **Resolution:** These differences likely stem from different simulation protocols (Gilbert burst model vs. binomial random deletion vs. iterative deletion), different definitions of "no interpolation" (retaining NaNs vs. deleting points), and different window lengths (2 min vs. 5 min). The conservative consensus is that **RMSSD does not benefit from complex interpolation**, and simple approaches (OR or nearest-neighbor) are preferable.

### Contradiction 2: Best Interpolation Method for SDNN and Frequency Domain
- **Morelli et al. (2019)** advocate **quadratic interpolation on time** as the best overall method for SDNN and frequency-domain features.
- **Benchekroun et al. (2023)** advocate **Pchip interpolation** as the best overall.
- **Cajal et al. (2022)** advocate **non-linear (Hermite) gap-filling** for frequency domain.
- **Resolution:** All three are non-linear, shape-preserving methods. The differences in performance are likely small and dataset-dependent. The key actionable insight is to use a **non-linear interpolation on timestamps**, not linear or nearest-neighbor on durations.

### Unresolved Question 1: Ultra-Short-Term Windows
- The reviewed studies used 2-minute (Cajal et al.) or 5-minute (Morelli et al., Benchekroun et al.) windows. There is limited evidence on how gap-filling performs in **ultra-short-term** windows (< 1 minute), which are increasingly common in wearable applications. Poincaré plot reliability in ultra-short segments has been demonstrated (Cajal et al., 2022, citing others), but the interaction with missing data is underexplored.

### Unresolved Question 2: Physiological Non-Stationarity During Gaps
- All interpolation methods assume that the underlying heart rate is approximately constant or smoothly varying across the gap. If the gap occurs during a rapid autonomic transition (e.g., standing up, startle response), no interpolation method can recover the true beat times. The literature does not provide clear guidance on how to detect or handle such non-stationary gaps.

### Unresolved Question 3: PPG vs. ECG Specificity
- Most studies used ECG as the gold standard and simulated missing data by deleting R-peaks. Real PPG-derived pulse rate variability (PRV) has different artifact patterns (e.g., motion artifacts causing both missed beats and false positives). The literature on correction methods specifically validated for PPG PRV is thinner than for ECG HRV.

---

## Limitations of This Synthesis

1. **Source verification method:** The `paper-exists` tool (Semantic Scholar) was non-functional during this session. Papers were verified by direct access to PubMed Central, ScienceDirect, and publisher websites. This is a robust verification but does not follow the prescribed workflow.

2. **Number of primary sources:** While 12 sources were reviewed, only 4--5 are primary empirical studies directly comparing correction methods. The 1996 Task Force paper provides foundational definitions but does not address modern wearable-specific missing-data scenarios.

3. **Temporal scope:** The most relevant papers are from 2019--2023, reflecting the recent surge in wearable-device research. Earlier foundational work (Kim et al., 2007; Clifford & Tarassenko, 2005) focused on smaller artifacts and ectopic beats rather than large motion-induced gaps.

4. **No direct mathematical proof:** The explanation for why RMSSD is more sensitive than meanNN is based on the mathematical definition and empirical observations. No reviewed paper provides a formal mathematical proof of relative sensitivity in the presence of filtered outliers.

5. **Heterogeneous simulation protocols:** The primary studies used different simulation methods (Gilbert burst model, binomial random deletion, iterative deletion), different window lengths, and different error metrics (relative error, MAPE, RMSE). This makes direct numerical comparison across studies challenging.

6. **PPG-specificity gap:** The user's application processes both ECG and PPG. The literature on PPG PRV correction is less mature than ECG HRV correction. The Apple Watch dataset in Cajal et al. (2022) is one of the few real PPG validations.

---

## References

1. Cajal, D., Hernando, D., Lazaro, J., Laguna, P., Gil, E., & Bailon, R. (2022). Effects of Missing Data on Heart Rate Variability Metrics. *Sensors*, 22(15), 5774. https://doi.org/10.3390/s22155774

2. Morelli, D., Rossi, A., Cairo, M., & Clifton, D. A. (2019). Analysis of the Impact of Interpolation Methods of Missing RR-intervals Caused by Motion Artifacts on HRV Features Estimations. *Sensors*, 19(14), 3163. https://doi.org/10.3390/s19143163

3. Benchekroun, M., Chevallier, B., Zalc, V., Istrate, D., Lenne, D., & Vera, N. (2023). The Impact of Missing Data on Heart Rate Variability Features: A Comparative Study of Interpolation Methods for Ambulatory Health Monitoring. *IRBM*, 44(4), 100776. https://doi.org/10.1016/j.irbm.2023.100776

4. Bourdillon, N., Yazdani, S., Vesin, J.-M., Schmitt, L., & Millet, G. P. (2022). RMSSD Is More Sensitive to Artifacts Than Frequency-Domain Parameters: Implication in Athletes' Monitoring. *Journal of Sports Science and Medicine*, 21, 260--266. https://doi.org/10.52082/jssm.2022.260

5. Malik, M., et al. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *European Heart Journal*, 17, 354--381. https://doi.org/10.1111/j.1542-474X.1996.tb00275.x

6. Kim, K. K., Lim, Y. G., Kim, J. S., & Park, K. S. (2007). Effect of missing RR-interval data on heart rate variability analysis in the time domain. *Physiological Measurement*, 28, 1485--1494. https://doi.org/10.1088/0967-3334/28/12/003

7. Kim, K. K., Kim, J. S., Lim, Y. G., & Park, K. S. (2009). The effect of missing RR-interval data on heart rate variability analysis in the frequency domain. *Physiological Measurement*, 30, 1039--1050. https://doi.org/10.1088/0967-3334/30/10/005

8. Clifford, G. D., & Tarassenko, L. (2005). Quantifying errors in spectral estimates of HRV due to beat replacement and resampling. *IEEE Transactions on Biomedical Engineering*, 52(4), 630--638. https://doi.org/10.1109/TBME.2005.844028

9. Peltola, M. A. (2012). Role of editing of R--R intervals in the analysis of heart rate variability. *Frontiers in Physiology*, 3, 148. https://doi.org/10.3389/fphys.2012.00148

10. Kubios HRV Software. (2024). Preprocessing of HRV data. Kubios Blog. https://www.kubios.com/blog/preprocessing-of-hrv-data

11. Altini, M. (2023). Data quality for heart rate variability (HRV) measurement. Marco Altini's Substack. https://marcoaltini.substack.com/p/data-quality-for-heart-rate-variability

12. Mateo, J., & Laguna, P. (2000). Analysis of heart rate variability using the IPFM model with time-varying threshold: Application to ectopic beat detection. *IEEE Transactions on Biomedical Engineering*. (Cited in Cajal et al., 2022)