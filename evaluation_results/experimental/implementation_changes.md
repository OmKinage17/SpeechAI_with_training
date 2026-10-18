# SpeechAI Experimental Implementation Changes & Architecture Inventory

## 1. Overview of Changes

In accordance with user instructions, all modifications and improvements have been implemented in complete isolation under `backend/experimental/` and `backend/tests/`. No production source code, APIs, or existing benchmark artifacts were modified or overwritten.

---

## 2. Inventory of Created Files

| File Path | Module Role | Lines of Code | Key Capabilities |
|---|---|:---:|---|
| [`backend/experimental/__init__.py`](file:///e:/SpeechPrj/backend/experimental/__init__.py) | Package Entrypoint | 24 | Exports `ExperimentalConfig`, `ExperimentalASREngine`, `AcousticFeatureExtractor`, `HybridStutterDetector`, `ContextAwareFillerDetector`, `DevelopmentEvaluationHarness`. |
| [`backend/experimental/config.py`](file:///e:/SpeechPrj/backend/experimental/config.py) | Hyperparameter Configuration | 99 | Strongly typed dataclasses for ASR, Acoustic, Stutter, and Filler configurations with default parameters grounded in empirical findings. |
| [`backend/experimental/asr_engine.py`](file:///e:/SpeechPrj/backend/experimental/asr_engine.py) | Resilient Speech Recognition | 144 | Bounded temperature decoding (`0.0, 0.2`), thread-pool per-clip timeout protection (`25.0s`), `static-ffmpeg` path initialization, structured status reporting (`SUCCESS`, `TIMEOUT`, `ERROR`). |
| [`backend/experimental/acoustic_features.py`](file:///e:/SpeechPrj/backend/experimental/acoustic_features.py) | Digital Signal Processing | 196 | Frame-level RMS energy, bounded noise floor estimation ($0.003\text{--}0.040$), pYIN fundamental frequency ($F_0$) tracking, spectral flux stability, formant schwa/nasal resonance verification. |
| [`backend/experimental/stutter_detector.py`](file:///e:/SpeechPrj/backend/experimental/stutter_detector.py) | Multimodal Stutter Detection | 273 | Lexical sound/syllable/word repetitions, acoustic prolongation detection (>320ms steady pitch), acoustic block detection (>400ms tense silence + plosive burst), continuous confidence scoring $[0.0, 1.0]$, calibrated uncertainty boundaries. |
| [`backend/experimental/filler_detector.py`](file:///e:/SpeechPrj/backend/experimental/filler_detector.py) | Context-Aware Filler Detection | 260 | Unconditional vocal hesitations (`um`, `uh`), grammatical contextual exclusions for `"so"`, `"well"`, and `"like"`, consecutive repeated marker detection (`like like`), formant-verified acoustic filled pause analysis. |
| [`backend/experimental/eval_harness.py`](file:///e:/SpeechPrj/backend/experimental/eval_harness.py) | Development Evaluation Engine | 263 | Comparative development evaluation harness computing 3-class dominant metrics, binary stutter PR-F1, binary filler PR-F1, confusion matrices, latency/RTF distribution, and ASR WER against baseline. |
| [`backend/experimental/run_dev_evaluation.py`](file:///e:/SpeechPrj/backend/experimental/run_dev_evaluation.py) | Development Benchmark Runner | 120 | Executes the evaluation harness on a balanced 30-clip development set (10 UCLASS, 10 FluencyBank, 10 Libri_Stutter) from pilot holdout partitions. |
| [`backend/tests/test_experimental_pipeline.py`](file:///e:/SpeechPrj/backend/tests/test_experimental_pipeline.py) | Automated Unit Test Suite | 250 | 18 automated unit and integration tests covering contextual gating, acoustic formant verification, synthetic prolongations, acoustic blocks, clean speech rejection, and ASR timeout handling. |

---

## 3. Detailed Architectural Enhancements

### A. Automatic Speech Recognition (ASR)
- **Elimination of Fallback Tail Loops**:
  - *Baseline*: Default Whisper temperature schedule `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`. On disfluent audio, repetitive compression failures triggered up to 6 passes of beam search and DTW on CPU, causing single-clip latency spikes up to 1,939s (32.3 minutes).
  - *Experimental*: Constrained to `(0.0, 0.2)`. Greedy decoding (`0.0`) handles clear utterances; at most one temperature fallback (`0.2`) is permitted.
- **Strict Per-Clip Timeout Protection**:
  - *Implementation*: `ThreadPoolExecutor` worker with a configurable timeout (`25.0s`). If Whisper hangs or loops, execution is halted, latency is recorded, and status is flagged as `TIMEOUT` without crashing the application.
- **Environment Integration**:
  - Integrated `static_ffmpeg.add_paths()` within `asr_engine.py` to ensure reliable `ffmpeg` binary resolution across all Windows host environments.

### B. Acoustic Digital Signal Processing
- **Noise Floor and Dynamic Thresholding**:
  - Implemented dynamic noise floor estimation using the quietest 15th percentile of frames, bounded between $0.003$ and $0.040$. This prevents near-zero division on synthetic/digital silence and prevents threshold blowout on continuously loud audio.
- **Pitch and Voicing Analysis**:
  - Configured `librosa.pyin` with $F_{0\min} = 75\text{ Hz}$ and $F_{0\max} = 400\text{ Hz}$ with 32ms frame windows.
- **Formant Resonance Verification**:
  - Analyzed power spectrum in the central vowel/nasal formant band ($300\text{--}1800\text{ Hz}$) relative to total energy. Requires central band ratio $\ge 0.28$, zero crossing rate (ZCR) $< 0.25$, and low-frequency rumble ($< 100\text{ Hz}$) $< 0.40$ to verify vocalic hesitation vs. unvoiced breath noise or microphone rumble.

### C. Hybrid Stutter Detection
- **Multi-Level Lexical Repetitions**:
  - Detects triplicate prefixes (`w-w-what`), duplicate prefixes (`w-what`), broken syllable fragments (`b- because`), and word repetitions (`the the`).
- **Acoustic Prolongation**:
  - Flags sustained voiced segments $\ge 320\text{ms}$ with stable pitch ($< 2.0\text{ semitones}$ standard deviation) and low spectral flux ($< 0.35$).
- **Acoustic Silent Blocks**:
  - Detects silent gaps $\ge 400\text{ms}$ between words (evaluating inner gap energy with 30ms margin from word boundaries) followed by a sudden plosive onset burst ($\ge 2.5\times$ gap energy and $\ge \text{vocal\_threshold}$).
- **Continuous Confidence & Uncertainty**:
  - Computes calibrated continuous confidence $c \in [0.0, 1.0]$.
  - Assigns explicit classifications: `CONFIDENT_DISFLUENT` ($c \ge 0.60$), `UNCERTAIN_BORDERLINE` ($0.35 \le c < 0.60$), and `CONFIDENT_CLEAN` ($c < 0.35$).

### D. Context-Aware Filler Detection
- **Grammatical Exclusions for Discourse Markers**:
  - Excludes standard grammatical usages of `"so"` (degree adverbs, conjunctions, pro-forms), `"well"` (adverbs of manner, adjectives, well-known, well-done), and `"like"` (verbs, comparisons) using compiled regex patterns.
- **Context Gating**:
  - Candidate discourse markers must be adjacent to an acoustic pause boundary ($\ge 200\text{ms}$) or appear as consecutive repetitions (`like like`, `so so`).
- **Formant-Verified Acoustic Filled Pauses**:
  - Replaced crude energy-gap counting with spectral resonance verification.

---

## 4. Preservation of Baseline & System Integrity

The following existing files and directories have been verified as **strictly unchanged**:
- `backend/main.py`: Untouched. Production FastAPI endpoints and scoring logic preserved.
- `backend/run_main_benchmark.py`: Untouched.
- `backend/video_module/*`: All video, vision, audio, scoring, and windows modules untouched.
- `evaluation_results/pilot_manifest.json`: Untouched.
- `evaluation_results/pilot_predictions.json`: Untouched.
- `evaluation_results/main_benchmark/*`: All 750 raw predictions, manifest, summary report, and audit artifacts untouched.
