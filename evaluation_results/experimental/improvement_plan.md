# SpeechAI Experimental Improvement Plan: Empirical Diagnosis & Modular Architecture

## 1. Executive Summary & Problem Diagnosis

Following the empirical audit of the 750-clip SpeechAI benchmark (`E:\SpeechPrj\evaluation_results\main_benchmark\`), three critical failure modes were identified in the baseline production pipeline:

1. **Severe Stutter False Negatives (Recall: 29.94%)**:
   - *Verified Cause*: The baseline `detect_repetitions()` function relied strictly on text regex patterns (e.g. `w-w-what`, `the the`). It contained **zero acoustic digital signal processing**. When Whisper smoothed out disfluent speech into grammatically normalized English text, phonetic prolongations (held vowels/consonants) and silent articulatory blocks were completely lost, causing 70.06% of stutter events to be missed.
2. **High Filler False Positives (Precision: 41.36%, 173 False Alarms)**:
   - *Verified Cause*: The baseline `detect_fillers()` unconditionally flagged every instance of words like `"so"`, `"well"`, and `"like"` as fillers regardless of grammatical context (e.g., *"I don't think so"*, *"as well"*, *"I like coffee"*). Furthermore, baseline acoustic hesitation detection flagged unvoiced breath turbulence and background noise gaps without formant resonance validation.
3. **Extreme Latency Tail Spikes (9 clips = 63.92% of Total Execution Time)**:
   - *Verified Cause*: Whisper's default temperature fallback ladder (`0.0, 0.2, 0.4, 0.6, 0.8, 1.0`) was repeatedly triggered on severely disfluent audio, causing up to 6 beam-search passes with Dynamic Time Warping (DTW) per clip, causing latency spikes up to 1,939 seconds (32.3 minutes).

To address these empirical bottlenecks without disrupting production services or benchmark artifacts, an isolated, modular experimental architecture was created under `backend/experimental/`.

---

## 2. Experimental Architecture Overview

The experimental architecture is partitioned into five specialized modules:

```mermaid
flowchart TD
    Audio[Audio Input .wav / .flac] --> ASR[ASR Engine: Bounded Whisper (0.0, 0.2) + 25s Timeout]
    Audio --> Acoustic[Acoustic Feature Extractor: Librosa / NumPy]
    
    ASR --> Transcript[Transcript + Timestamped Word Alignment]
    
    Transcript --> StutterDet[Hybrid Stutter Detector]
    Acoustic --> StutterDet
    
    Transcript --> FillerDet[Context-Aware Filler Detector]
    Acoustic --> FillerDet
    
    StutterDet --> StutterOutput[Stutter Events, Confidence & Classification]
    FillerDet --> FillerOutput[Filler Events, Confidence & Classification]
    
    StutterOutput --> EvalHarness[Development Evaluation Harness]
    FillerOutput --> EvalHarness
    Transcript --> EvalHarness
```

### Module Specifications

| Module | Source File | Responsibilities | Dependencies |
|---|---|---|---|
| **Configuration** | `config.py` | Centralized hyperparameter dataclasses for ASR, acoustics, stutter, and fillers | Standard Library (`dataclasses`) |
| **ASR Engine** | `asr_engine.py` | Safe, bounded Whisper inference with thread-pool timeout, static-ffmpeg integration, and error reporting | `openai-whisper`, `static-ffmpeg` |
| **Acoustic Features** | `acoustic_features.py` | Energy profiles, noise floors, pYIN pitch tracking, spectral flux, formant resonance ratios | `librosa`, `soundfile`, `numpy`, `scipy` |
| **Hybrid Stutter** | `stutter_detector.py` | Multi-level lexical repetition + acoustic prolongation + acoustic block detection + continuous confidence | `re`, `numpy` |
| **Contextual Filler** | `filler_detector.py` | Grammatically gated discourse markers + formant-verified filled pause analysis | `re`, `numpy` |
| **Eval Harness** | `eval_harness.py` | Side-by-side comparative benchmarking against audited baseline predictions | `numpy`, `jiwer` |

---

## 3. Detailed Component Designs

### 3.1 Bounded ASR Decoding & Latency Protection
- **Decoding Configuration**: Restricts decoding temperatures to `(0.0, 0.2)` instead of the default 6-step ladder. Greedy decoding (`0.0`) handles >95% of speech clips; at most one temperature fallback (`0.2`) is permitted if repetitive compression occurs.
- **Asynchronous Per-Clip Timeout**: Each audio clip is executed within a dedicated worker via `ThreadPoolExecutor` with a default 25.0s timeout limit. If execution exceeds 25.0s, the process gracefully aborts the clip, flags it as `TIMEOUT`, and logs latency without hanging the entire pipeline.
- **Structured Status Reporting**: Outputs explicit status tags (`SUCCESS`, `TIMEOUT`, `ERROR`) allowing downstream evaluation tools to distinguish inference timeouts from clean speech.

### 3.2 Acoustic Digital Signal Processing
- **Dynamic Energy Profiling**: Frame-level RMS energy calculated using 32ms frames (hop: 10ms). The noise floor is estimated from the lowest 15th percentile of frames, bounded between $0.003$ and $0.040$ to avoid near-zero division on digital silence or threshold blowout on continuous speech.
- **Articulatory Stability (Spectral Flux)**: Normalized spectral difference between consecutive frames. Steady-state prolonged vowels exhibit low spectral flux ($\le 0.35$).
- **pYIN Pitch & Voicing Tracking**: Extracts fundamental frequency ($F_0$) constrained to human vocal range ($75\text{--}400\text{ Hz}$). Prolonged vowels are verified by pitch stability ($\le 2.0\text{ semitones}$ standard deviation across $\ge 320\text{ms}$).
- **Formant Resonance Verification**: Validates filled pause candidates by computing power spectral distribution in central vowel / nasal resonance bands ($300\text{--}1800\text{ Hz}$). Requires central band energy ratio $\ge 0.28$, zero crossing rate (ZCR) $< 0.25$, and low-frequency rumble ($< 100\text{ Hz}$) $< 0.40$ to reject breathing and mic noise.

### 3.3 Hybrid Stutter Detection
- **Lexical Modality**:
  - Triplicate prefix repetitions (`w-w-what`, confidence = 0.95)
  - Duplicate prefix repetitions (`w-what`, confidence = 0.85)
  - Broken syllable fragments (`b- because`, confidence = 0.80)
  - Whole-word repetitions (`the the`, confidence = 0.85)
- **Acoustic Prolongation Modality**:
  - Sustained continuous voicing $\ge 320\text{ms}$ with low pitch variation ($< 2.0\text{ semitones}$) and low spectral flux ($< 0.35$).
- **Acoustic Block Modality**:
  - Silent articulatory gaps ($\ge 400\text{ms}$) with inner-gap energy $\le \max(0.03, \text{vocal\_threshold} \times 1.5)$ followed by an onset plosive energy release ($\ge 2.5\times$ gap energy and $\ge \text{vocal\_threshold}$).
- **Confidence Scoring & Uncertainty**:
  - Continuous score $c \in [0.0, 1.0]$ derived from multi-event evidence.
  - Decision threshold: $c \ge 0.60 \implies \text{CONFIDENT\_DISFLUENT}$; $0.35 \le c < 0.60 \implies \text{UNCERTAIN\_BORDERLINE}$; $c < 0.35 \implies \text{CONFIDENT\_CLEAN}$.

### 3.4 Context-Aware Filler Detection
- **Core Vocal Fillers (Unconditional)**:
  - `"um"`, `"uh"`, `"er"`, `"erm"`, `"ah"`, `"hmm"` and orthographic variations (`^u+m+h*$`, `^u+h+m*$`): High confidence ($0.90$).
- **Multi-Word Phrase Fillers**:
  - `"you know"`, `"i mean"`, `"sort of"`, `"kind of"`: Confidence ($0.80$).
- **Context-Gated Discourse Markers ("so", "well", "like")**:
  - Grammatical exclusions:
    - `"so"`: Degree adverbs (*"so good"*, *"so far"*), conjunctions (*"so that"*), pro-forms (*"think so"*, *"hope so"*), and idioms (*"or so"*).
    - `"well"`: Adverbs of manner (*"do well"*, *"as well"*), adjectives (*"getting well"*), nouns (*"water well"*), compound adjectives (*"well-known"*, *"well-done"*).
    - `"like"`: Verbs (*"I like"*, *"would like to"*), comparisons (*"looks like"*, *"feels like"*).
  - Positive filler conditions:
    - Preceded or followed by an acoustic pause boundary $\ge 200\text{ms}$.
    - Consecutive repetition (*"like like"*, *"so so"*).
    - Utterance-initial hesitation followed by core vocal filler (*"So... um"*).
- **Formant-Verified Acoustic Filled Pauses**:
  - Inter-word pauses ($0.25\text{--}2.0\text{s}$) verified via central formant schwa resonance ratio ($\ge 0.28$) and low ZCR ($< 0.25$).

---

## 4. Verification & Validation Framework

1. **Automated Unit Tests (`backend/tests/test_experimental_pipeline.py`)**:
   - 18 automated unit tests covering contextual false positive rejection, genuine vocal fillers, synthetic schwa resonance, unvoiced breath noise rejection, low rumble rejection, synthetic acoustic prolongations, acoustic blocks, clean speech rejection, and ASR timeout handling.
2. **Controlled Development Benchmark Set (30 clips)**:
   - Evaluated strictly on pilot/development partitions (`M_0030`, `FB_Ep_010`, Reader `103`).
   - Side-by-side comparison against audited baseline predictions from `pilot_predictions.json`.
   - Zero overlap with the final 750-clip holdout.

---

## 5. Scope & Safety Guarantees

- **No Overwriting of Existing Files**: `backend/main.py`, `backend/run_main_benchmark.py`, `evaluation_results/main_benchmark/` and pilot artifacts remain strictly unmodified.
- **No Holdout Contamination**: The final 750-clip benchmark holdout is preserved and was not touched or re-run during this phase.
- **No Automated Deployment**: All experimental code lives strictly in `backend/experimental/`.
