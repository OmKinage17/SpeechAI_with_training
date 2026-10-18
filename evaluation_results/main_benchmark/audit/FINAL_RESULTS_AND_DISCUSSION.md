# Results and Discussion

This section presents the empirical evaluation of the SpeechAI disfluency assessment architecture across 750 preflight-verified holdout audio clips. All reported metrics are independently reproducible and audited directly from the saved benchmark predictions and audit artifacts (`recomputed_metrics.json`, `recomputed_predictions_metrics.csv`, and `audit_validation.json`) located in `evaluation_results/main_benchmark/audit/`.

---

## 1. Experimental Setup and Evaluation Corpora

To evaluate generalizability across varying speech registers and recording conditions, the benchmark utilizes three distinct corpora with strict holdout isolation. All partitions enforce **zero speaker, episode, and reader overlap** with pilot or development data:

1. **UCLASS (Release One):** Clinical monologue recordings of individuals who stutter (Holdout speakers: `M_0078`, `M_0107`, `M_0121`, `M_0553`; $N = 250$). Total audio duration: 750.00 seconds (fixed 3.0-second clips).
2. **FluencyBank / SEP-28k:** Real-world conversational speech extracted from podcasts (Holdout episodes: `FB_Ep_016`, `FB_Ep_024`; $N = 250$; annotator consensus threshold $\ge 2$). Total audio duration: 750.00 seconds (fixed 3.0-second clips).
3. **Libri_Stutter:** Verbatim reading speech with semi-synthetic disfluency insertions on LibriSpeech recordings (Holdout readers: `1088`, `1334`, `1502`; $N = 250$). Total audio duration: 3,513.61 seconds (variable sentence lengths, mean: 14.05 seconds).

The total evaluation pool comprises **750 unique audio clips** representing **5,013.61 seconds (1.393 hours)** of speech. The pipeline executed with a **100% completion rate (750 / 750 successful runs)** without unhandled exceptions or data loss.

---

## 2. Multi-Class Dominant Disfluency Classification

To evaluate overall utterance-level categorization, each audio clip was assigned a single mutually exclusive ground-truth and predicted class following the clinical priority hierarchy:
$$\text{Stutter} \succ \text{Filler} \succ \text{Clean}$$
Under this hierarchy, an utterance containing both stuttering repetitions and filled pauses is classified as `Stutter`. An utterance is classified as `Filler` only if it contains interjections without repetitions or prolongations, and as `Clean` only in the complete absence of disfluency annotations.

### Table 1: Dominant Three-Class Confusion Matrices (Rows: Ground Truth, Columns: Predictions)

$$\begin{array}{r|ccc|c|ccc|c}
\hline
& \multicolumn{4}{c|}{\textbf{UCLASS (Clinical Monologue)}} & \multicolumn{4}{c}{\textbf{FluencyBank (Conversational Podcasts)}} \\
\hline
\text{\textbf{Ground Truth \textbackslash\ Predicted}} & \textbf{Clean} & \textbf{Stutter} & \textbf{Filler} & \textbf{Support} & \textbf{Clean} & \textbf{Stutter} & \textbf{Filler} & \textbf{Support} \\
\hline
\textbf{Clean}   & 48 & 10 & 34 & 92  & 106 & 4  & 30 & 140 \\
\textbf{Stutter} & 47 & 40 & 42 & 129 & 29  & 7  & 11 & 47  \\
\textbf{Filler}  & 8  & 5  & 16 & 29  & 9   & 6  & 48 & 63  \\
\hline
\textbf{Predicted Total} & 103 & 55 & 92 & 250 & 144 & 17 & 89 & 250 \\
\hline
\textbf{Corpus Accuracy} & \multicolumn{4}{c|}{\mathbf{41.60\%}} & \multicolumn{4}{c}{\mathbf{64.40\%}} \\
\hline
\end{array}$$

$$\begin{array}{r|ccc|c|ccc|c}
\hline
& \multicolumn{4}{c|}{\textbf{Libri\_Stutter (Controlled Reading)}} & \multicolumn{4}{c}{\textbf{Aggregate Benchmark (N = 750)}} \\
\hline
\text{\textbf{Ground Truth \textbackslash\ Predicted}} & \textbf{Clean} & \textbf{Stutter} & \textbf{Filler} & \textbf{Support} & \textbf{Clean} & \textbf{Stutter} & \textbf{Filler} & \textbf{Support} \\
\hline
\textbf{Clean}   & 62  & 2  & 18 & 82  & 216 & 16  & 82  & 314 \\
\textbf{Stutter} & 81  & 56 & 31 & 168 & 157 & 103 & 84  & 344 \\
\textbf{Filler}  & 0   & 0  & 0  & 0   & 17  & 11  & 64  & 92  \\
\hline
\textbf{Predicted Total} & 143 & 58 & 49 & 250 & 390 & 130 & 230 & \mathbf{750} \\
\hline
\textbf{Corpus Accuracy} & \multicolumn{4}{c|}{\mathbf{47.20\%}} & \multicolumn{4}{c}{\mathbf{51.07\%}} \\
\hline
\end{array}$$

### Table 2: Dominant Three-Class Classification Performance

| Corpus | Target Class | Ground-Truth Support | TP | FP | FN | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCLASS** | Clean | 92 | 48 | 55 | 44 | — | 0.4660 | 0.5217 | 0.4923 |
| *(Overall Acc: 41.60%)* | Stutter | 129 | 40 | 15 | 89 | — | 0.7273 | 0.3101 | 0.4348 |
| | Filler | 29 | 16 | 76 | 13 | — | 0.1739 | 0.5517 | 0.2645 |
| | *Macro Average* | 250 | — | — | — | 0.4160 | 0.4557 | 0.4612 | 0.3972 |
| | *Weighted Average* | 250 | — | — | — | 0.4160 | 0.5669 | 0.4160 | 0.4362 |
| **FluencyBank** | Clean | 140 | 106 | 38 | 34 | — | 0.7361 | 0.7571 | 0.7465 |
| *(Overall Acc: 64.40%)* | Stutter | 47 | 7 | 10 | 40 | — | 0.4118 | 0.1489 | 0.2188 |
| | Filler | 63 | 48 | 41 | 15 | — | 0.5393 | 0.7619 | 0.6316 |
| | *Macro Average* | 250 | — | — | — | 0.6440 | 0.5624 | 0.5560 | 0.5323 |
| | *Weighted Average* | 250 | — | — | — | 0.6440 | 0.6255 | 0.6440 | 0.6183 |
| **Libri_Stutter** | Clean | 82 | 62 | 81 | 20 | — | 0.4336 | 0.7561 | 0.5511 |
| *(Overall Acc: 47.20%)* | Stutter | 168 | 56 | 2 | 112 | — | 0.9655 | 0.3333 | 0.4956 |
| | Filler | 0 | 0 | 49 | 0 | — | 0.0000 | Undefined* | Undefined* |
| | *Macro Average* | 250 | — | — | — | 0.4720 | 0.6995 | 0.5447 | 0.5233 |
| | *Weighted Average* | 250 | — | — | — | 0.4720 | 0.7910 | 0.4720 | 0.5138 |
| **Aggregate** | **Clean** | **314** | **216** | **174** | **98** | — | **0.5538** | **0.6879** | **0.6136** |
| **(N = 750)** | **Stutter** | **344** | **103** | **27** | **241** | — | **0.7923** | **0.2994** | **0.4346** |
| *(Overall Acc: 51.07%)* | **Filler** | **92** | **64** | **166** | **28** | — | **0.2783** | **0.6957** | **0.3975**$^\dagger$ |
| | *Macro Average* | **750** | — | — | — | **0.5107** | **0.5415** | **0.5610** | **0.4819** |
| | *Weighted Average* | **750** | — | — | — | **0.5107** | **0.6294** | **0.5107** | **0.5050** |
| | *Micro Average* | **750** | — | — | — | **0.5107** | **0.5107** | **0.5107** | **0.5107** |

*\*Note on Libri_Stutter Filler Support:* Because Libri_Stutter contains zero ground-truth filler annotations (support = 0), recall and F1-score are mathematically $\frac{0}{0}$ (undefined). Reporting $0.0$ via default automated metric libraries is misleading and corrected here to `Undefined (Support = 0)`.  
$^\dagger$*Note on Aggregate Filler F1 Rounding:* The raw unrounded harmonic mean ($\frac{128}{322}$) evaluates to `0.397515...` ($\mathbf{0.3975}$). Calculating F1 from intermediate four-decimal rounded precision (0.2783) and recall (0.6957) yields `0.397563...` ($\mathbf{0.3976}$).

---

## 3. Clip-Level Binary Presence Detection

In clinical practice, speech disfluencies frequently co-occur; an individual may exhibit a sound repetition immediately preceded by an audible pause or filler. Dominant single-class assignment artificially suppresses secondary disfluencies. Therefore, we independently evaluate **clip-level binary presence detection** ($y \in \{0, 1\}$), where the target is whether a clip contains at least one instance of the given disfluency category regardless of co-occurring events.

### Table 3: Clip-Level Binary Detection Performance Across Corpora

| Evaluation Category | Corpus | Positive Support | TP | FP | FN | TN | Binary Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stutter Detection** | UCLASS | 129 | 40 | 15 | 89 | 106 | 0.5840 | 0.7273 | 0.3101 | 0.4348 |
| *(Sound Repetitions,* | FluencyBank | 47 | 7 | 10 | 40 | 193 | 0.8000 | 0.4118 | 0.1489 | 0.2188 |
| *Word Repetitions,* | Libri_Stutter | 168 | 56 | 2 | 112 | 80 | 0.5440 | **0.9655** | 0.3333 | **0.4956** |
| *Prolongations)* | **Aggregate** | **344** | **103** | **27** | **241** | **379** | **0.6427** | **0.7923** | **0.2994** | **0.4346** |
| **Filler Detection** | UCLASS | 84 | 61 | 70 | 23 | 96 | 0.6280 | 0.4656 | 0.7262 | 0.5674 |
| *(Filled Pauses,* | FluencyBank | 72 | 61 | 39 | 11 | 139 | 0.8000 | 0.6100 | **0.8472** | **0.7093** |
| *Interjections)* | Libri_Stutter | 0 | 0 | 64 | 0 | 186 | 0.7440 | 0.0000 | Undefined* | Undefined* |
| | **Aggregate** | **156** | **122** | **173** | **34** | **421** | **0.7240** | **0.4136** | **0.7821** | **0.5410** |

### Relationship Between Dominant and Binary Metrics
1. **Mathematical Identity of Stutter Detection:** In dominant-class evaluation, Stutter occupies the highest priority tier ($\text{Stutter} \succ \text{Filler} \succ \text{Clean}$). Consequently, any clip containing a repetition or prolongation was assigned ground-truth `Stutter`, and any clip with predicted stammers was assigned predicted `Stutter`. The binary contingency table for Stutter detection ($\text{TP}=103, \text{FP}=27, \text{FN}=241$) is mathematically identical to the dominant class contingency table.
2. **Divergence of Filler Detection:** In dominant classification, 64 clips containing both a stutter and a filler were assigned to `Stutter`, leaving only 92 dominant-filler ground-truth clips. In binary evaluation, all 156 clips with filler annotations are evaluated. Binary filler recall increases to **78.21% (122 / 156)** with an F1-score of **0.5410**, demonstrating that the hybrid acoustic-lexical detector captures conversational hesitations effectively even when co-occurring with overt stutters.

---

## 4. Automatic Speech Recognition Accuracy Under Stuttering (WER)

ASR transcription performance was evaluated on the **Libri_Stutter** benchmark partition ($N = 250$), where verbatim ground-truth reading texts are available. Word Error Rate (WER) is evaluated across two distinct formulations:

1. **Macro-Average WER (Clip-Level Mean):**
   $$\text{WER}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} \frac{S_i + D_i + I_i}{N_{\text{ref}, i}} = \mathbf{0.2493} \quad (24.93\%)$$
   This metric weights each clip equally regardless of duration, reflecting the average user utterance experience.
2. **Corpus-Level Micro WER (Pooled Word Error Rate):**
   $$\text{WER}_{\text{micro}} = \frac{\sum_{i=1}^{N} (S_i + D_i + I_i)}{\sum_{i=1}^{N} N_{\text{ref}, i}} = \frac{2,083}{4,773} = \mathbf{0.4364} \quad (43.64\%)$$
   This metric aggregates total substitution ($S$), deletion ($D$), and insertion ($I$) edits across the entire concatenated vocabulary (4,773 reference words).

### Interpretation of ASR Degradation
A macro WER of 24.93% demonstrates that Whisper `base` retains intelligible semantic grounding on short disfluent passages. However, the substantially higher corpus micro WER (43.64%) highlights that longer reading sentences with severe multi-token repetitions and prolonged silent blocks suffer severe insertion and deletion cascades.

---

## 5. Latency, Throughput, and Real-Time Factor (RTF)

Inference timings were measured end-to-end on host CPU (Intel x86_64 architecture, FP32 execution) from audio load through Whisper feature extraction, acoustic DSP analysis, regex parsing, and database serialization. Model loading overhead (1.11 seconds) occurred once during initialization and is excluded from per-clip timings.

### Table 4: Measured Hardware Execution Profile ($N = 750$ Clips)

| Performance Metric | Measured Value | Formal Definition / Academic Significance |
| :--- | :---: | :--- |
| **Total Audio Processed** | 5,013.61 s (1.393 hrs) | Combined duration of all 750 holdout speech utterances. |
| **Total Benchmark Wall-Clock** | 150.92 min (9,055.16 s) | Sum of all per-clip processing times ($\sum \text{latency}_i$). |
| **System Throughput** | 4.97 clips / min | Total completed clips divided by total execution minutes. |
| **Median Latency** | **2.521 seconds** | 50th percentile of clip processing times ($P_{50}$). |
| **Mean Latency** | **12.074 seconds** | Total wall-clock time divided by 750 clips. |
| **Standard Deviation** | 91.333 seconds | Reflects extreme positive skew from decoding outliers. |
| **95th Percentile ($P_{95}$)** | 16.211 seconds | Upper bound latency for 95% of typical utterances. |
| **Minimum Latency** | 0.882 seconds | Short clean audio clip processing. |
| **Maximum Latency** | 1,917.945 seconds (~31.97 min) | Multi-pass decoding cascade on clip `1088-134315-0007.flac`. |
| **Median RTF** | **0.792$\times$** | $\text{Median}(\text{latency}_i / \text{duration}_i)$; faster than real-time for typical clips. |
| **Aggregate Micro RTF** | **1.806$\times$** | $\sum \text{latency}_i / \sum \text{duration}_i = 9055.16\text{s} / 5013.61\text{s}$. |
| **Macro-Average RTF** | **3.303$\times$** | $\frac{1}{N}\sum (\text{latency}_i / \text{duration}_i)$; unweighted arithmetic mean. |

### Analysis of the Nine Extreme-Latency Outliers
A critical architectural finding is the dramatic divergence between median latency (2.521s) and mean latency (12.074s). Audit analysis reveals that this discrepancy is driven by **exactly nine outlier clips (1.2% of the dataset)**:
- In these nine clips (e.g., `1088-134315-0007.flac` at 1,917.95s; `FluencyBank_024_10.wav` at 975.98s; `FluencyBank_016_19.wav` at 883.49s), processing times exceeded 100 seconds per clip.
- Together, these nine clips consumed **5,787.72 seconds (96.46 minutes)**, representing **63.92% of the entire benchmark wall-clock time**.
- Conversely, the remaining **741 clips (98.8%)** processed in **3,267.44 seconds**, yielding a mean latency of **4.409 seconds/clip** (matching the pilot benchmark's measured throughput of 4.395s/clip).
- *Root Cause:* When Whisper encounters severe phonetic repetitions, prolonged silences, or acoustic block postures, its repetition penalty and compression-ratio thresholds fail. This triggers the internal temperature fallback ladder (`temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`), forcing up to six successive beam-search decoding passes on CPU.

---

## 6. Discussion: The "Stutter-Smoothing" Paradox and Recall Constraints

A primary scientific contribution of this benchmark is characterizing the fundamental trade-off between modern self-supervised ASR backbones and downstream disfluency detection:

```
[Audio: Severe Acoustic Stutter / Prolongation / Block]
                     │
                     ▼
       ┌───────────────────────────┐
       │ Whisper Transformer ASR   │
       │ (Trained on 680k hrs text)│
       └─────────────┬─────────────┘
                     │ Inductive Language Model Bias
                     │ (Silently normalizes acoustic stutters into clean text)
                     ▼
         Transcribed Output: "The presentation went well."
         (Overt acoustic repetition erased from transcript)
                     │
                     ▼
       ┌───────────────────────────┐
       │ Downstream Regex Detector │
       │ Checks for "th-th-the"    │
       └─────────────┬─────────────┘
                     │
                     ▼
         Prediction: Clean / Fluent  ===> FALSE NEGATIVE (Missed Stutter)
```

1. **High Stutter Precision (79.23% Aggregate, 96.55% Reading):** When SpeechAI flags a stutter event, it is highly reliable. False positive alarms are rare (only 27 across 750 clips), demonstrating that lexical repetition patterns and acoustic pause heuristics accurately reflect genuine stuttering when orthographically transcribed.
2. **Low Stutter Recall (29.94% Aggregate, 14.89% Conversational):** The critical vulnerability is sensitivity: **70.06% of true stuttered clips were missed (241 false negatives out of 344 clips)**. Whisper's autoregressive decoder is optimized for maximum word accuracy; its strong language model prior actively "heals" and normalizes disfluent acoustic fragments into standard words. Unless a repetition is severe enough to force explicit orthographic hyphenation, text-based rules cannot detect it.
3. **Clinical Risk of Missed Disfluencies:** In an assessment context, reporting high precision without disclosing low recall would be irresponsible. An assessment tool that misses over 70% of stuttered utterances would severely underestimate an individual's stuttering severity, potentially denying needed support. SpeechAI must therefore never be used as a standalone diagnostic evaluator.

---

## 7. Scope of the System: Implemented Functionality vs. Future Work

To prevent misapplication, the current technological maturity of SpeechAI must be precisely delineated:

### Implemented Functionality (Evaluated in this Study)
- Multi-corpus ingestion pipeline with audio integrity validation.
- Whisper `base` transcription with prompt conditioning for conversational disfluencies.
- Hybrid acoustic DSP (pitch stability and spectral flux tracking) combined with lexical pattern matching for filled-pause detection.
- Lexical and syllable repetition regex analysis for overt stutter detection.
- Asynchronous session report generation with longitudinal tracking.

### Proposed Future Work (Not Implemented in Current System)
- **Acoustic-Loss Fine-Tuning:** Fine-tuning Whisper or Conformer encoders with Connectionist Temporal Classification (CTC) loss without language model smoothing to prevent normalization of repeated phones.
- **Dedicated Frame-Level Classifiers:** Integrating acoustic models (e.g., fine-tuned Wav2Vec 2.0 or HuBERT) directly on raw spectrograms to classify silent blocks and prolongations independent of text transcription.
- **Fixed-Temperature Greedy Decoding:** Disabling the multi-pass temperature fallback ladder to cap per-clip CPU latency under 5.0 seconds.
- **Hardware Acceleration:** GPU deployment (CUDA FP16) to achieve true streaming real-time factor ($\text{RTF} < 0.2\times$).
- **Direct Sub-type Classification:** Multi-head neural decoders capable of distinguishing sound repetitions, word repetitions, prolongations, and blocks as distinct continuous streams.

---

## 8. Limitations

1. **Corpus Heterogeneity:** The three datasets represent vastly different recording conditions. UCLASS comprises clinical monologues from children and adults; FluencyBank contains edited, multi-speaker conversational podcasts; and Libri_Stutter contains semi-synthetic acoustic insertions over clean audiobooks. Cross-corpus performance variance reflects these differing acoustic domains.
2. **Zero Filler Ground Truth in Libri_Stutter:** Because Libri_Stutter focuses exclusively on stutter insertions, it contains zero annotated fillers. The 49 clips flagged as containing fillers reflect conversational hesitation patterns or false alarms, but recall cannot be evaluated on this corpus.
3. **Unavailable Sub-Type Specific Predictions:** The current production pipeline emits an aggregated `pred_stammers` integer count. It does not provide isolated, independent predictions for sound repetitions versus word repetitions versus prolongations. Consequently, granular receiver operating characteristic (ROC) curves for individual stutter sub-types cannot be generated without modifying the core model architecture.
4. **Hardware and Latency Constraints:** Although the median processing time is 2.52 seconds ($\text{RTF} = 0.792\times$), the aggregate micro RTF is $1.806\times$ and the macro RTF is $3.303\times$. The system cannot support real-time streaming feedback on CPU hardware.
5. **Lack of Clinical Diagnostic Validation:** SpeechAI is an experimental screening and asynchronous session-review tool. It has not undergone formal clinical trials, does not calculate standard clinical severity indexes such as the Stuttering Severity Instrument (SSI-4), and cannot replace a qualified Speech-Language Pathologist (SLP).

---

## 9. Conclusion

The 750-clip benchmark establishes baseline performance for the hybrid Whisper and acoustic DSP architecture under strict holdout validation. SpeechAI demonstrates strong precision for overt stutter detection (79.23%) and high sensitivity for conversational filled pauses (78.21%), while preserving semantic transcription with a macro WER of 24.93%. However, due to ASR language-model smoothing, stutter recall is constrained to 29.94%, and temperature fallback loops introduce severe latency tails on CPU. These findings confirm that SpeechAI is effective as an experimental, asynchronous screening aid, but highlight that clinical diagnostic deployment will require direct acoustic frame-level modeling to overcome the inherent smoothing of sequence-to-sequence ASR.
