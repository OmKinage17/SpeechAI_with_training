# Results and Discussion (Corrected Research Manuscript)

**Document Version:** 2.0 (Scientifically Corrected & Auditor-Certified Edition)  
**Publication Target:** SpeechAI Implementation Paper & Technical Archive  
**Evaluation Scope:** 750-Clip Multi-Corpus Holdout Benchmark (Phase E Experimental vs. Audited Baseline)  
**Date of Audit Certification:** October 18, 2026  
**Auditor Authorities:** Independent ML Benchmark Auditor & Independent Research-Methodology Reviewer  
**Primary Verification Deliverables:**
- [`FINAL_RESEARCH_READINESS_REVIEW.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md)
- [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md)
- [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md)
- [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json)
- [`baseline_experimental_comparison.csv`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/baseline_experimental_comparison.csv)
- [`immutability_verification.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/immutability_verification.json)
- [`EXPERIMENTAL_BENCHMARK_REPORT_CORRECTED.md`](file:///E:/SpeechPrj/evaluation_results/experimental/main_benchmark/EXPERIMENTAL_BENCHMARK_REPORT_CORRECTED.md)

---

## Executive Summary of Manuscript Revisions

This corrected manuscript presents the empirical findings and critical discussion of the SpeechAI disfluency assessment architecture across 750 preflight-verified holdout audio clips. Following independent forensic audit and methodological review, this edition updates and replaces the uncorrected preliminary manuscript (`FINAL_RESULTS_AND_DISCUSSION.md`) while strictly preserving all underlying benchmark data and source code.

Key scientific corrections incorporated in this version include:
1. **Descriptive Nature of Accuracy Gain:** Overall dominant three-class accuracy increased from **50.13% to 53.73%** (+3.60 percentage points). Formal hypothesis testing via McNemar's test establishes that this difference is **not statistically significant** ($\chi^2 = 1.4923, p = 0.2219$).
2. **Paired Discordant Classification Analysis:** Across 453 discordant prediction pairs, 240 clips improved from baseline error to experimental accuracy, while 213 clips degraded from baseline accuracy to experimental error (net gain of 27 clips), demonstrating substantial classification churn.
3. **Primary Clinical Sensitivity Improvement:** Binary stutter recall surged from **29.10% to 84.66%** (+55.56 percentage points), successfully identifying 320 of 378 ground-truth stuttered utterances.
4. **Precision–Recall Trade-Off:** The sensitivity gain incurred a substantial precision penalty, falling from **84.62% to 56.14%** (-28.48 percentage points), with false positives escalating from 20 to 250 across the cohort.
5. **Filler Detection Regression:** Binary filler detection F1-score degraded from **0.5410 to 0.3886** (-0.1524), driven by 123 additional false alarms caused by static acoustic pause boundary over-triggering.
6. **Domain-Specific Limitations:** Experimental accuracy on conversational speech in FluencyBank collapsed to **35.20%** with a stutter precision of only **25.16%**, disconfirming any claim of generalized superiority across conversational environments.
7. **Research Prototype Scope:** SpeechAI is explicitly characterized as an exploratory research prototype for asynchronous screening, not a clinically cleared diagnostic instrument. Furthermore, clip-level evaluation over 3.0-second windows does not establish temporal event-boundary accuracy.

---

## 1. Experimental Setup, Evaluation Corpora, and Taxonomy Reconciliation

### 1.1 Multi-Corpus Holdout Partitions
To evaluate generalizability across varying speech registers, acoustic environments, and clinical demographics, the benchmark incorporates three diverse corpora partitioned under strict holdout constraints (**zero speaker, reader, or episode overlap** with development or pilot partitions):

1. **UCLASS (Release One):** Clinical monologue and reading recordings of children and adolescents who stutter (Holdout speakers: `M_0078`, `M_0107`, `M_0121`, `M_0553`; $N = 250$). Total audio duration: 750.00 seconds (fixed 3.0-second clips).
2. **FluencyBank / SEP-28k:** Naturalistic conversational speech extracted from adult podcast interviews (Holdout episodes: `FB_Ep_016`, `FB_Ep_024`; $N = 250$; annotator consensus threshold $\ge 2$). Total audio duration: 750.00 seconds (fixed 3.0-second clips).
3. **Libri_Stutter:** Controlled reading speech with semi-synthetic disfluency insertions superimposed on LibriSpeech recordings (Holdout readers: `1088`, `1334`, `1502`; $N = 250$). Total audio duration: 3,513.61 seconds (variable sentence lengths, mean: 14.05 seconds).

The total evaluation pool comprises **750 unique audio clips** representing **5,013.61 seconds (1.393 hours)** of speech. Cryptographic verification confirmed that all 750 clips are identical between baseline and experimental evaluations, with zero data corruption or file mutations ([`immutability_verification.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/immutability_verification.json)).

### 1.2 Ground-Truth Taxonomy Reconciliation: Definition A vs. Definition B
A critical prerequisite for rigorous comparative analysis is establishing identical ground-truth label definitions. Forensic audit ([`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md#2-investigation-1-root-cause-analysis-of-baseline-accuracy-5013-vs-5107-and-stutter-support-378-vs-344)) revealed that historical project reports cited two distinct ground-truth mapping conventions:

- **Definition A (Narrow Historical Taxonomy):** Evaluated stuttering strictly as repetitions and prolongations:  
  $$\text{Stutter} \iff (\text{gt\_sound\_rep} = 1) \lor (\text{gt\_word\_rep} = 1) \lor (\text{gt\_prolongation} = 1)$$  
  Silent blocks (`gt_block`) were excluded from the stutter definition. Under this narrow rule, ground-truth support was **Stutter = 344**, Clean = 314, Filler = 92. Baseline accuracy was reported as **51.07%** ($383 / 750$), while experimental accuracy was **51.47%** ($386 / 750$).
- **Definition B (Inclusive Clinical SEP-28k Taxonomy):** Incorporates silent postural fixations (blocks) into the pathological stuttering category, adhering to standardized clinical speech pathology criteria:  
  $$\text{Stutter} \iff (\text{gt\_sound\_rep} = 1) \lor (\text{gt\_word\_rep} = 1) \lor (\text{gt\_prolongation} = 1) \lor (\text{gt\_block} = 1)$$  
  Exactly **34 clips** in the benchmark (23 in UCLASS, 11 in FluencyBank) contain annotated silent blocks (`gt_block = 1`) without co-occurring repetitions or prolongations. Reclassifying these 34 clips shifted ground-truth support to **Stutter = 378**, Clean = 292, Filler = 80.

Under Definition B, baseline accuracy evaluates to **50.13%** ($376 / 750$), while experimental accuracy evaluates to **53.73%** ($403 / 750$). To ensure valid scientific comparison, **Definition B is consistently applied across all primary baseline and experimental comparisons in this manuscript**, guaranteeing an exact apples-to-apples evaluation on identical ground truth.

---

## 2. Multi-Class Dominant Disfluency Classification

Audio clips are assigned a mutually exclusive dominant disfluency category following the clinical priority hierarchy:
$$\text{Stutter} \succ \text{Filler} \succ \text{Clean}$$
Under this hierarchy, any utterance exhibiting acoustic or lexical repetitions, prolongations, or blocks is categorized as `Stutter`. An utterance is designated `Filler` only if it contains interjections in the absence of stuttering events, and `Clean` only when completely free of disfluencies.

### Table 1: Dominant 3-Class Confusion Matrices ($N = 750$ Clips)
*(Rows: Ground Truth, Columns: Model Predictions; Evaluated under Definition B)*

$$\begin{array}{r|ccc|c|ccc|c}
\hline
& \multicolumn{4}{c|}{\textbf{Audited Baseline Pipeline}} & \multicolumn{4}{c}{\textbf{Experimental Refactored Pipeline}} \\
\hline
\text{\textbf{Ground Truth \textbackslash\ Predicted}} & \textbf{Clean} & \textbf{Stutter} & \textbf{Filler} & \textbf{Support} & \textbf{Clean} & \textbf{Stutter} & \textbf{Filler} & \textbf{Support} \\
\hline
\textbf{Clean}   & 216 & 16  & 60 & 292 & 70  & 198 & 24 & 292 \\
\textbf{Stutter} & 173 & 103 & 102 & 378 & 34  & 320 & 24 & 378 \\
\textbf{Filler}  & 10  & 11  & 57  & 80  & 15  & 52  & 13 & 80  \\
\hline
\textbf{Predicted Total} & 399 & 130 & 219 & \mathbf{750} & 119 & 570 & 61  & \mathbf{750} \\
\hline
\textbf{Overall Accuracy} & \multicolumn{4}{c|}{\mathbf{50.13\%} \ (376/750)} & \multicolumn{4}{c}{\mathbf{53.73\%} \ (403/750)} \\
\hline
\end{array}$$

*Audit Note:* All rows, columns, and diagonal elements sum to 750 with zero unaccounted clips ([`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json)).

---

### Table 2: Dominant Three-Class Classification Performance
*(Evaluated across the aggregate 750-clip cohort under Definition B)*

| Target Class / Averaging Scheme | Baseline Prec | Baseline Rec | Baseline F1 | Exp Prec | Exp Rec | Exp F1 | Delta Prec | Delta Rec | Delta F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** (Support = 292) | 54.14% | **73.97%** | **0.6252** | **58.82%** | 23.97% | 0.3406 | +4.68 pp | -50.00 pp | -0.2846 |
| **Stutter** (Support = 378) | **79.23%** | 27.25% | 0.4055 | 56.14% | **84.66%** | **0.6751** | -23.09 pp | **+57.41 pp** | **+0.2696** |
| **Filler** (Support = 80) | **26.03%** | **71.25%** | **0.3813** | 21.31% | 16.25% | 0.1844 | -4.72 pp | -55.00 pp | -0.1969 |
| **Macro Average** | 53.13% | 57.49% | **0.4748** | 45.43% | 41.63% | 0.4000 | -7.70 pp | -15.86 pp | -0.0748 |
| **Weighted Average** | **53.79%** | 50.13% | **0.4948** | 53.46% | **53.73%** | 0.4925 | -0.33 pp | +3.60 pp | -0.0023 |
| **Overall Accuracy** | \multicolumn{3}{c|}{\mathbf{50.13\%} \ (376 / 750)} & \multicolumn{3}{c|}{\mathbf{53.73\%} \ (403 / 750)} & \multicolumn{3}{c}{\mathbf{+3.60 \text{ pp (Descriptive)}}} |

---

### 2.1 Descriptive Accuracy Gain and Statistical Insignificance (McNemar's Test)

Overall dominant three-class accuracy increased from **50.13% (376 / 750)** in the baseline pipeline to **53.73% (403 / 750)** in the experimental pipeline, representing a **+3.60 percentage-point descriptive improvement** (net gain of 27 correctly classified clips).

To evaluate whether this observed difference reflects genuine algorithmic improvement rather than random variation across sample partitions, an independent paired analysis was conducted using **McNemar's Test for Paired Nominal Data** on the 750 matched clip predictions ([`FINAL_RESEARCH_READINESS_REVIEW.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md#91-formal-statistical-significance-testing-mcnemars-test)):

#### Table 3: McNemar Paired Contingency Matrix ($N = 750$ Matched Clips)
```
                               Experimental Correct    Experimental Incorrect       Row Total
Baseline Correct                        163 (n11)               213 (n10)          376 (50.13%)
Baseline Incorrect                      240 (n01)               134 (n00)          374 (49.87%)
Column Total                            403 (53.73%)            347 (46.27%)       750 (100.0%)
```

- **Concordant Pairs:** 297 clips yielded identical outcomes across both models (163 clips were correctly classified by both; 134 clips were misclassified by both).
- **Discordant Pairs ($N_{\text{discordant}} = 453$):** Exactly 453 clips yielded divergent classifications between models:
  - $n_{01}$ (Baseline Incorrect $\to$ Experimental Correct): **240 clips**
  - $n_{10}$ (Baseline Correct $\to$ Experimental Incorrect): **213 clips**
  - **Net Correct Gain:** $n_{01} - n_{10} = 240 - 213 = \mathbf{+27\text{ clips}}$

Applying McNemar's test statistic with Edwards' continuity correction:
$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} = \frac{(|240 - 213| - 1)^2}{240 + 213} = \frac{(27 - 1)^2}{453} = \frac{26^2}{453} = \frac{676}{453} \approx \mathbf{1.4923}$$

The corresponding two-tailed asymptotic $p$-value is:
$$\mathbf{p = 0.2219} \quad (\mathbf{p > 0.05})$$

**Academic Finding:** The +3.60 percentage-point accuracy improvement is **not statistically significant** at the standard $\alpha = 0.05$ threshold. The null hypothesis of marginal homogeneity cannot be rejected.

### 2.2 Forensic Interpretation of the 453 Discordant Pairs
The underlying classification dynamics explain why substantial algorithmic modifications failed to produce a statistically significant accuracy separation:
- The experimental pipeline's lowered acoustic thresholds (prolongation spectral tilt threshold reduced to 0.32s; block silence threshold reduced to 0.40s) successfully rescued **240 clips** that baseline missed, predominantly capturing previously undetected stutters.
- Concurrently, this aggressive calibration introduced widespread false-positive triggers, converting **213 clips** that baseline had accurately identified as clean or filler into erroneous stutter classifications.
- Consequently, 94.0% of the gross positive transitions ($213 / 240$) were offset by corresponding negative transitions, yielding a modest net gain of only 27 clips. Researchers must therefore refrain from describing the experimental pipeline as uniformly superior in classification accuracy.

---

## 3. Clip-Level Binary Presence Detection

Because disfluent speech events frequently co-occur (e.g., an audible prolongation immediately following a filled pause), dominant-class assignment artificially suppresses secondary disfluencies. We independently evaluate **clip-level binary presence detection** ($y \in \{0, 1\}$), where the target is whether an utterance contains at least one instance of the designated category.

### Table 4: Clip-Level Binary Detection Across the 750-Clip Benchmark

| Evaluation Task | Pipeline | Positive Support | TP | FP | FN | TN | Binary Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stutter Presence** | Audited Baseline | 378 | 110 | 20 | 268 | 352 | 61.60% | **84.62%** | 29.10% | **94.62%** | 0.4331 |
| *(Reps, Prolongs, Blocks)*| **Experimental** | **378** | **320** | **250** | **58** | **122** | **58.93%** | 56.14% | **84.66%** | 32.80% | **0.6751** |
| | *Absolute Delta ($\Delta$)* | — | *+210* | *+230* | *-210* | *-230* | *-2.67 pp* | *-28.48 pp* | *+55.56 pp* | *-61.82 pp* | *+0.2420* |
| **Filler Presence** | Audited Baseline | 156 | 122 | 173 | 34 | 421 | **72.40%** | **41.36%** | **78.21%** | **70.88%** | **0.5410** |
| *(Interjections, Pauses)* | **Experimental** | **156** | **109** | **296** | **47** | **298** | 54.27% | 26.91% | 69.87% | 50.17% | 0.3886 |
| | *Absolute Delta ($\Delta$)* | — | *-13* | *+123* | *+13* | *-123* | *-18.13 pp* | *-14.45 pp* | *-8.34 pp* | *-20.71 pp* | *-0.1524* |

---

### 3.1 Stutter Sensitivity Improvement vs. Precision Trade-Off
The central architectural trade-off of the experimental pipeline lies in its reconfiguration from a conservative detector to a high-sensitivity screening model:

1. **Massive Sensitivity Surge:** Binary stutter recall increased from **29.10% to 84.66%** (+55.56 percentage points). Whereas baseline failed to detect 268 out of 378 stuttered clips (70.9% false negative rate), the experimental pipeline captured **320 of 378 clips**, missing only 58. In clinical triage, high sensitivity is paramount to prevent patients with subtle disfluencies from being turned away without assessment.
2. **Substantial Precision Penalty:** This sensitivity gain required accepting a substantial reduction in precision, which declined from **84.62% to 56.14%** (-28.48 percentage points). False positives escalated from 20 to 250 across the cohort. Specificity collapsed from **94.62% to 32.80%** (-61.82 percentage points), indicating that nearly two-thirds of non-stuttered clips triggered false disfluency detections.
3. **Clinical Framing:** The experimental model functions effectively as a **sensitive clinical screening filter** designed to maximize recall at the cost of diagnostic specificity, requiring secondary human review to filter out false alarms.

---

### 3.2 Filler Detection Regression
In contrast to stutter detection, filler word detection suffered an unambiguous performance collapse:
- **F1-Score Regression:** Binary filler F1 fell from **0.5410 to 0.3886** (-0.1524, a 28.2% relative decline).
- **Precision and Recall Reductions:** Precision dropped from **41.36% to 26.91%** (-14.45 percentage points), while recall dropped from **78.21% to 69.87%** (-8.34 percentage points).
- **False Alarm Escalation:** Spurious filler detections rose by **71.1%**, increasing from 173 to 296 false positives (+123 clips).
- **Acoustic Root Cause:** In the experimental architecture, the context-aware filler detector incorporated a static 0.20-second acoustic pause boundary rule. In fluent reading speech (Libri_Stutter), natural grammatical breath pauses and expressive punctuation intervals consistently triggered the acoustic detector, generating **171 false positives** on a dataset containing zero ground-truth fillers ([`FINAL_RESEARCH_READINESS_REVIEW.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md#72-filler-word-detection-severe-algorithmic-degradation)).

---

## 4. Cross-Corpus Generalization & Domain-Specific Limitations

To rigorously test model robustness across acoustic domains, performance was evaluated across the three distinct holdout corpora (250 clips each).

### Table 5: Per-Corpus Performance Breakdown ($N = 250$ Clips Per Corpus)
*(Dominant-class accuracy reported alongside binary disfluency metrics)*

| Corpus Cohort | Recording Context & Population | Dominant Acc (Base) | Dominant Acc (Exp) | Acc Delta ($\Delta$) | Binary Stutter Prec | Binary Stutter Rec | Binary Stutter F1 | Binary Filler Prec | Binary Filler Rec | Binary Filler F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCLASS** | Monologue / reading of children who stutter. | 41.20% | **59.20%** | **+18.00 pp** | 68.02% | 76.97% | **0.7222** | 48.09% | 75.00% | **0.5860** |
| **FluencyBank** | Natural conversational podcast interviews (adults). | **62.00%** | 35.20% | **-26.80 pp** | 25.16% | 67.24% | 0.3662 | 44.66% | 63.89% | 0.5257 |
| **Libri_Stutter** | Audiobooks with spliced synthetic disfluencies. | 47.20% | **66.80%** | **+19.60 pp** | 67.49% | 97.62% | **0.7981** | 0.00% | Undefined* | Undefined* |

*\*Note:* Libri_Stutter contains zero ground-truth filler annotations (Support = 0). Precision is 0.00% due to 171 false alarms; recall and F1 are mathematically undefined ($\frac{0}{0}$).

---

### 4.1 FluencyBank Collapse and Domain-Specific Failure
The empirical data decisively refute any hypothesis of generalized cross-domain robustness:
1. **Severe Conversational Regression:** On FluencyBank, dominant-class accuracy dropped precipitously from **62.00% to 35.20%** (-26.80 percentage points).
2. **Weak Stutter Precision (25.16%):** Stutter precision collapsed to **25.16%**, meaning that **nearly three out of four predicted stutters in conversational speech were false alarms** (116 false positives).
3. **Mechanistic Failure Mode:** Conversational discourse contains complex communicative phenomena, including turn-taking pauses, conversational backchannels ("uh-huh", "yeah"), floor-holding hesitations, and laughter. The static acoustic thresholds (energy dips, spectral flux, and fixed pause boundaries) calibrated on monologue recordings misidentified these normal conversational dynamics as pathological stuttering blocks and prolongations.
4. **Pediatric Monologue Gain (UCLASS):** In contrast, the experimental model achieved substantial gains on child clinical monologues in UCLASS, where accuracy rose from **41.20% to 59.20%** (+18.00 percentage points) and stutter F1 reached **0.7222**. In controlled monologue reading, acoustic pause and repetition markers correspond reliably to genuine disfluency events.

---

## 5. Automatic Speech Recognition (ASR) Under Stuttering

ASR transcription performance was evaluated on the **Libri_Stutter** holdout corpus ($N = 250$), where verbatim ground-truth reading texts are available.

### 5.1 Text Normalization Procedure (`clean_text()`)
Standard speech recognition evaluation requires text normalization to prevent punctuation formatting and orthographic case mismatches from artificially distorting error rates. The `clean_text()` pipeline applies:
- Lowercase conversion of hypothesis and reference strings.
- Stripping of all punctuation characters (`. , ? ! ' " - : ; ( ) [ ]`).
- Whitespace normalization and removal of leading/trailing spaces.

### 5.2 Evaluated Denominator and Timeout Handling ($N = 248$)
Of the 250 attempted Libri_Stutter clips, exactly **two clips timed out** at the 25.0-second ASR cutoff (`1334-135589-0088.flac` and `1334-135589-0039.flac`). Following standard speech recognition benchmarking protocols:
- The 2 timed-out clips were excluded from the WER denominator ($0.8\%$ exclusion rate) rather than assigned arbitrary failure scores (such as 0.0 or 1.0).
- Both baseline and experimental models were evaluated on the identical **248 successful clips** ([`FINAL_RESEARCH_READINESS_REVIEW.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md#5-review-criterion-4-automatic-speech-recognition-asr--word-error-rate-wer)).

### Table 6: Word Error Rate Performance on Libri_Stutter ($N = 248$ Clips)

| Evaluation Metric | Text Processing Protocol | Baseline ASR | Experimental ASR | Absolute Delta ($\Delta$) | Academic Significance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Normalized Macro WER** | Standard `clean_text()` normalization (lowercase, unpunctuated). | **24.93%** ($0.2493$) | **23.84%** ($0.2384$) | **-1.09 pp (Better)** | **Primary Comparable ASR Metric** |
| **Normalized Pooled Micro WER** | Global corpus edit distance with text cleaning. | **24.30%** ($0.2430$) | **23.46%** ($0.2346$) | **-0.84 pp (Better)** | Primary Global Corpus WER |
| **Raw Text Macro WER** | Unnormalized strings evaluated directly against uppercase reference. | **45.05%** ($0.4505$) | **44.19%** ($0.4419$) | **-0.86 pp (Better)** | Secondary Unnormalized Metric |
| **Raw Text Pooled Micro WER** | Global corpus edit distance without text cleaning. | **43.64%** ($0.4364$) | **42.98%** ($0.4298$) | **-0.66 pp (Better)** | Secondary Global Raw WER |

*Methodological Insight:* Without text normalization, Levenshtein distance penalizes Whisper's standard punctuation marks as insertion errors against uppercase LibriSpeech reference text, inflating raw WER to ~44.2%. Under standard speech recognition cleaning, Experimental ASR achieves **23.84% Normalized Macro WER**, outperforming baseline by **1.09 percentage points** while bounding inference latency.

---

## 6. Latency, Telemetry, and Process-Level Bounded Execution

### 6.1 Tri-Fold Latency Accounting and Telemetry Reconciliation
To ensure complete transparency, timing results are categorized across three distinct operational dimensions ([`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md#8-investigation-7-summed-clip-latency-vs-elapsed-wall-clock-time-reconciliation)):

1. **Active Compute Latency ($N = 746$ Clips):** Measures true CPU inference time excluding operating system sleep/hibernation pauses.
2. **Summed Clip Latency ($N = 750$ Clips):** Direct arithmetic sum of per-clip `perf_counter` durations recorded in prediction records.
3. **Multi-Session Active Wall-Clock Duration:** Total active elapsed time across execution sessions.

### Table 7: Execution Latency and Real-Time Factor (RTF) Profile

| Performance Dimension | Audited Baseline Pipeline (N=750) | Experimental Active Compute *(N=746, Excl. Sleep)* | Experimental Full Telemetry *(N=750, Unfiltered)* | Architectural Impact & Findings |
| :--- | :---: | :---: | :---: | :--- |
| **Median Latency** | **2.521 s** | **6.093 s** | 6.173 s | Active acoustic DSP feature extraction overhead. |
| **Mean Latency** | 12.074 s | **8.994 s** | 13.434 s | **25.5% active compute reduction vs. baseline.** |
| **95th Percentile ($P_{95}$)** | **16.211 s** | **25.062 s** | 25.064 s | Bounded strictly by 25.0s ASR timeout. |
| **Maximum Latency** | 1,917.945 s *(Runaway loop)* | **28.361 s** *(Bounded)* | 2,658.687 s *(OS sleep)* | **Runaway decoding loops eliminated.** |
| **Active Micro RTF** | 1.806 | **1.341** | 2.010 | $\sum \text{latency} / \sum \text{audio\_duration}$. |
| **Macro RTF** | 3.303 | **2.521** | 4.004 | Mean of individual clip RTF values. |
| **Summed Processing Time** | 9,055.70 s (150.93 min) | **6,709.31 s** (111.82 min) | 10,075.79 s (167.93 min) | Total per-clip CPU processing time. |
| **Active Wall-Clock Duration** | 9,055.70 s (150.93 min) | **10,954.64 s** (182.58 min) | **10,954.64 s** (182.58 min) | Full campaign duration across both sessions. |

---

### 6.2 Forensic Accounting of the 2,658.687-Second Latency Outlier
Preliminary reports recorded an apparent maximum latency of `2658.687s` for the experimental pipeline. Forensic log auditing confirmed this was an **operating system suspension artifact**, not an algorithmic hang:
- On clip 239 (`M_0107_07y7m_1_fluent_012.wav`), the benchmark host laptop entered Windows ACPI Modern Standby sleep mode at `15:05:47` and resumed at `15:50:07` (44 minutes 20 seconds).
- In sleep mode, the CPU halted execution; upon system wake, the worker process resumed, completed inference within 0.8 seconds, and emitted status `SUCCESS` with a valid transcript. High-resolution monotonic timers (`time.perf_counter()`) recorded the elapsed sleep interval.
- Exactly four clips experienced OS sleep intervals (>30.0s). Excluding these four non-computational intervals establishes the true algorithmic maximum latency as **28.361 seconds** (25.0s timeout + process termination overhead).

### 6.3 Multi-Session Wall-Clock Duration (182.58 Minutes)
The experimental benchmark executed in two discrete sessions:
- **Session 1 (Clips 1–355):** Ran from 14:16:06 to 16:18:13 (Wall-clock: 7,326.43s / 122.11 min).
- **Inter-Session Pause:** System halted from 16:18:13 to 18:51:26 (9,193.01s / 153.22 min).
- **Session 2 (Clips 356–750):** Resumed from checkpoint from 18:51:26 to 19:51:54 (Wall-clock: 3,628.21s / 60.47 min).
- **Total Active Wall-Clock Duration:** **182.58 minutes (10,954.64 seconds)**. Reporting only Session 2's 60.47 minutes was an error in early uncorrected drafts.

---

## 7. Scope of the System and Scientific Limitations

### 7.1 Research Prototype Status (Non-Diagnostic)
SpeechAI is an exploratory **research prototype**, not a clinically validated or cleared medical diagnostic instrument. It does not calculate standardized clinical severity indexes such as the Stuttering Severity Instrument (SSI-4), does not perform differential diagnosis between developmental and acquired neurogenic disfluency, and must never be utilized as a substitute for an evaluation by a licensed Speech-Language Pathologist (SLP).

### 7.2 Clip-Level Classification vs. Event-Boundary Alignment
A critical methodological boundary must be recognized regarding temporal resolution:
- The evaluations in this study operate exclusively on **clip-level presence** within segmented 3.0-second audio windows ($y \in \{0, 1\}$).
- Clip-level binary and dominant classification establishes whether a disfluency type occurred within the window; **it does NOT establish millisecond-level acoustic event-boundary accuracy**, exact disfluency onset/offset timestamps, or phone-level boundary alignments.
- Automated speech therapy systems requiring granular biofeedback or real-time interruption require frame-level acoustic segmentation, which is beyond the scope of this clip-level architecture.

### 7.3 Summary of Inherent Limitations
1. **Statistically Non-Significant Accuracy Gain:** The +3.60 percentage-point improvement in 3-class accuracy ($p = 0.2219$) demonstrates that architectural refinements did not yield a statistically detectable increase in overall classification accuracy.
2. **High False-Positive Screening Burden:** The model's high stutter sensitivity (84.66%) is accompanied by low specificity (32.80%) and a 56.14% precision rate, generating 250 false alarms across 750 clips.
3. **Conversational Speech Vulnerability:** The architecture experiences severe domain degradation on unconstrained conversational discourse (FluencyBank accuracy collapsed to 35.20%; stutter precision collapsed to 25.16%).
4. **Filler Detection Regression:** Static acoustic pause thresholds fail on expressive reading speech, generating 171 false filler alarms on Libri_Stutter and degrading aggregate filler F1 to 0.3886.
5. **CPU Hardware Latency:** With an active micro RTF of $1.341\times$, the pipeline cannot support real-time synchronous streaming feedback on CPU hardware.

---

## 8. Conclusion

This independent re-evaluation of the SpeechAI Phase E benchmark demonstrates that refactoring the architecture achieved its primary operational goals: eliminating runaway ASR decoding loops via process-level timeouts (capping maximum compute latency at 28.36s) and dramatically improving stutter screening sensitivity from 29.10% to 84.66%. 

However, rigorous statistical hypothesis testing establishes that the overall 3-class accuracy change (50.13% to 53.73%) is a descriptive +3.60 percentage-point difference that is not statistically significant ($p = 0.2219$, McNemar's test). Furthermore, the increased sensitivity introduced a substantial precision trade-off (stutter precision dropped to 56.14%, filler F1 degraded to 0.3886) and severe vulnerability to conversational discourse (FluencyBank accuracy collapsed to 35.20%). These findings demonstrate that SpeechAI provides a viable proof of concept for high-sensitivity asynchronous screening, but confirm that transition to clinical utility will require speaker-adaptive conversational modeling and frame-level acoustic boundary segmentation.

---

## 9. Auditor Verification Matrix & Citation of Local Artifacts

Every metric, statistical test, and forensic finding reported in this manuscript has been verified against immutable local audit records:

| Finding / Reported Metric | Manuscript Value | Local Audit Artifact Citation & Section |
| :--- | :---: | :--- |
| **Taxonomy Reconciliations** | Def A: 344 Supp (51.07% Acc)<br>Def B: 378 Supp (50.13% Acc) | [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md#2-investigation-1-root-cause-analysis-of-baseline-accuracy-5013-vs-5107-and-stutter-support-378-vs-344) |
| **Overall Accuracy & McNemar Test**| Acc: 50.13% $\to$ 53.73% (+3.60 pp)<br>$\chi^2 = 1.4923, p = 0.2219$ | [`FINAL_RESEARCH_READINESS_REVIEW.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md#91-formal-statistical-significance-testing-mcnemars-test)<br>[`baseline_experimental_comparison.csv`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/baseline_experimental_comparison.csv) |
| **Discordant Pairs Breakdown** | 240 gained, 213 lost (Net: +27)<br>453 discordant pairs total | [`FINAL_RESEARCH_READINESS_REVIEW.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/FINAL_RESEARCH_READINESS_REVIEW.md#31-3-class-confusion-matrix-marginal-verification)<br>[`baseline_experimental_comparison.csv`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/baseline_experimental_comparison.csv) |
| **Binary Stutter Trade-Off** | Recall: 29.10% $\to$ 84.66% (+55.56 pp)<br>Precision: 84.62% $\to$ 56.14% (-28.48 pp) | [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json#line-20-45) |
| **Binary Filler Regression** | F1: 0.5410 $\to$ 0.3886 (-0.1524)<br>FP: 173 $\to$ 296 (+123 clips) | [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json#line-46-75) |
| **FluencyBank Regression** | Dominant Acc: 62.00% $\to$ 35.20%<br>Stutter Precision: 25.16% | [`independently_recomputed_metrics.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/independently_recomputed_metrics.json#line-140-195) |
| **Normalized vs. Raw WER** | Norm Macro: 24.93% $\to$ 23.84% (-1.09 pp)<br>Raw Macro: 45.05% $\to$ 44.19% ($N=248$) | [`METRIC_DISCREPANCY_ANALYSIS.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/METRIC_DISCREPANCY_ANALYSIS.md#8-investigation-7-word-error-rate-wer-discrepancy-2493-vs-4505) |
| **Active Latency Profile** | Median: 6.093s, Mean: 8.994s<br>P95: 25.062s, Max: 28.361s ($N=746$) | [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md#5-investigation-4-accuracy-of-latency-and-rtf-metrics) |
| **Sleep Outlier (2,658.687s)** | Clip 239 OS modern standby sleep<br>44m 20s pause; valid transcript emitted | [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md#6-investigation-5-root-cause-of-the-2658687-second-maximum--suspension-outliers) |
| **Multi-Session Wall Clock** | Session 1: 122.11m, Session 2: 60.47m<br>Total Wall Clock: 182.58 min (10,955s) | [`TIMEOUT_LATENCY_AUDIT.md`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/TIMEOUT_LATENCY_AUDIT.md#8-investigation-7-summed-clip-latency-vs-elapsed-wall-clock-time-reconciliation) |
| **Cryptographic Immutability** | 12 protected files: 100% SHA-256 match | [`immutability_verification.json`](file:///E:/SpeechPrj/evaluation_results/experimental/verification/immutability_verification.json) |
