# SpeechAI Benchmark Analytics Dashboard

A dedicated, research-grade analytics dashboard for visualizing and analyzing the complete SpeechAI benchmark results from the 120-clip evaluation across all 7 project parameters.

> **Architecture Note:** This dashboard is completely isolated in `benchmark-dashboard/` and runs on a dedicated port (`http://localhost:4173`). It does not modify, replace, or interfere with the original SpeechAI application, its models, detector thresholds, predictions, or benchmark results.

---

## 🚀 Quick Start & Launch

### 1. Prerequisites
- **Node.js** v18+ (tested on Node v26.8.2)
- **npm** v9+ (tested on npm 11.19.1)

### 2. Installation
```bash
cd E:\SpeechPrj\benchmark-dashboard
npm install
```

### 3. Launch Dashboard
```bash
# Production Preview (Pre-built, recommended):
npm run preview

# Or Development Mode with Hot Module Replacement:
npm run dev
```
The dashboard will open automatically on:
**[http://localhost:4173](http://localhost:4173)**

### 4. Build for Production
```bash
npm run build
```

### 5. Run Data Integrity Verification Test
```bash
node tests/verify_dashboard_integrity.js
```
*Executes an automated 34-point research audit confirming 100% agreement between the dashboard data layer and the official benchmark audit reports.*

---

## 📊 Dashboard Architecture & Pages

The application is structured into 11 distinct research views accessible via the responsive sidebar:

| Page / Route | Parameter / Scope | Key Visualizations & Research Metrics |
| :--- | :--- | :--- |
| **Page A: Executive Overview** | All 7 Parameters ($N=120$) | 8 KPI Cards, 3×3 Multiclass Confusion Matrix, Baseline vs Final Comparison (+5.84% Acc, +0.0323 Stutter F1), Per-Dataset Stutter Bar Chart, Structured 7-Parameter Summary, Dataset Distribution. |
| **Page B: Stuttering Detection** | Parameter 1 ($N=120$) | Binary Confusion Matrix (TP=43, FP=15, FN=24, TN=38), P/R/F1/Spec Radar/Bar, Subtype Counts (156 Reps, 56 Prolongations, 26 Blocks), Gain Attribution Breakdown (+2 TP Timeouts, +1 TP Algorithmic), Per-Clip Error Inspector. |
| **Page C: Filler Detection** | Parameter 2 ($N=120$) | Dual-Task Evaluation Clarity Banner: Event-Level Detection (F1: 0.6966, Rec: 93.94%) vs Dominant-Class Classification (F1: 0.5424), Libri Negative Control Verification (0.0% FPR), Per-Dataset Breakdown. |
| **Page D: Clean-Speech Detection** | Parameter 3 ($N=120$) | Clean Confusion Matrix (TP=12, FP=12, FN=20, TN=76), Specificity: 86.36%, Precision: 50.00%, Recall: 37.50%, F1: 0.4286, Per-Dataset Specificity Bar Chart, False Clean vs Missed Clean Inspector. |
| **Page E: Speech-Rate Estimation** | Parameter 4 ($N=40$ Gold) | Scatter Plot with $y=x$ Identity Line, Residual Plot (Mean Bias: +11.84 WPM), Error Histogram, MAE: 17.84 WPM, RMSE: 22.15 WPM, Pearson $r$: 0.7547, Spearman $\rho$: 0.6914, 3s Window Bias Disclosure. |
| **Page F: Pause & Block Detection** | Parameter 5 ($N=120$ / $N=80$) | Dual-Section Layout: Pauses (200 silence events, histogram, mean 0.94s) + Clinical Blocks (Scoped UCLASS+FB $N=80$ CM: TP=0, FP=4, FN=6, TN=70, Spec: 94.59%, F1: 0.0000; Libri marked "Annotation Unavailable"; 6-clip FN audit). |
| **Page G: ASR Evaluation** | Parameter 6 ($N=40$ Gold) | WER: 23.94% & CER: 15.35% (299 edits / 1,249 reference words), WER Distribution Histogram, Latency Table, 0 Timeouts Confirmation, Hypothesis vs Gold Reference Diff Viewer, UCLASS/FB Scientific Disclosure. |
| **Page H: Coaching Engine** | Parameter 7 ($N=120$) | 100.0% Rule-Engine Consistency (120/120 clips), Mean Feedback: 1.74 tips/clip, Category Trigger Thresholds, Feedback Density Distribution, Sample Advice Table, Software Logic vs Clinical Efficacy Disclosure. |
| **Dataset Comparison** | 3-Way Cross-Corpus Matrix | Side-by-side 3-column comparative view (UCLASS vs FluencyBank vs Libri_Stutter) across all parameters with explicit annotation coverage tags, grouped bar charts, and per-corpus confusion matrices. |
| **Per-Clip Explorer** | Search & Audit ($N=120$) | Searchable, filterable 120-clip explorer with filters for dataset, true class, pred class, correctness, and error category, single-clip detail drawer modal, and CSV export. |
| **About & Limitations** | Methodology & Boundaries | Full documentation of dataset composition, 4-tier measurement taxonomy, fixed 3.0s window constraints, acoustic block blindspot, and engineering screening vs clinical diagnostic boundaries. |

---

## 🎛️ Global Controls & Filtering

At the top of every dashboard view:
1. **Dataset Selector:**
   - `Combined (All 3)` — All 120 clips
   - `UCLASS` — 40 clinical monologue & reading clips
   - `FluencyBank` — 40 spontaneous clinical interview clips
   - `Libri_Stutter` — 40 read speech audiobook clips with verbatim gold references
2. **Comparison Mode:**
   - `Combined Results` — Normal filtered dashboard mode
   - `Side-by-Side (3-Way)` — Direct 3-column cross-corpus comparative matrix
3. **Data Export:**
   - `Export CSV` — Exports filtered or full clip data with complete ground-truth labels and model predictions.
4. **Live Clip Count Badge:**
   - Dynamically reflects $N=120$ or $N=40$ based on the active dataset filter.

---

## 🛡️ Research Rigor & Epistemological Boundaries

1. **No Silent Negative Assumptions:**
   - Block detection ground truth is strictly annotated for UCLASS and FluencyBank ($N=80$). In Libri_Stutter ($N=40$), where block annotation was omitted from the schema, the dashboard displays `"Not evaluated — ground truth unavailable"` rather than assuming negative labels.
2. **Gold-Standard Scoping:**
   - ASR WER (23.94%) and Speech Rate regression metrics ($r = 0.7547$) are computed solely on Libri_Stutter ($N=40$), where authoritative verbatim book texts exist.
3. **Dual-Task Filler Clarity:**
   - Event-level filler detection ($F_1 = 0.6966$, Recall: $93.94\%$) is strictly decoupled from dominant-class classification ($F_1 = 0.5424$).
4. **Deterministic Audit vs Clinical Validation:**
   - The 100% score for Parameter 7 is prominently disclosed as deterministic rule execution consistency, not therapeutic efficacy.

---

## 📁 Project Directory Layout

```
benchmark-dashboard/
├── dist/                             # Pre-compiled production bundle
├── node_modules/
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── ConfusionMatrix.jsx       # Heatmap confusion matrix component
│   │   ├── Header.jsx                # Global filter bar & dataset switcher
│   │   ├── KpiCard.jsx               # Metric KPI cards with tooltips & deltas
│   │   ├── Sidebar.jsx               # Research navigation sidebar
│   │   └── StatusBadge.jsx           # Epistemological validation tier badges
│   ├── data/
│   │   ├── benchmark_unified_data.json # Reconciled 120-clip unified dataset
│   │   └── ... (original benchmark JSON/CSV copies)
│   ├── pages/
│   │   ├── ClipExplorer.jsx          # Per-clip inspector & CSV export
│   │   ├── DatasetComparison.jsx     # 3-way comparative matrix
│   │   ├── ExecutiveOverview.jsx     # Page A: High-level executive summary
│   │   ├── MethodologyLimitations.jsx# Methodology, taxonomy & boundaries
│   │   ├── Parameter1Stuttering.jsx  # Page B: Stuttering detection
│   │   ├── Parameter2Fillers.jsx     # Page C: Filler detection
│   │   ├── Parameter3CleanSpeech.jsx # Page D: Clean speech detection
│   │   ├── Parameter4SpeechRate.jsx  # Page E: Speech rate estimation
│   │   ├── Parameter5PauseBlock.jsx  # Page F: Pause & block analysis
│   │   ├── Parameter6ASR.jsx         # Page G: ASR WER / CER analysis
│   │   └── Parameter7Coaching.jsx    # Page H: Coaching rule audit
│   ├── utils/
│   │   └── metrics.js                # Dynamic research metrics recalculation engine
│   ├── App.jsx                       # Main application shell & global state
│   ├── index.css                     # Tailwind CSS & design tokens
│   └── main.jsx                      # Vite entrypoint
├── tests/
│   └── verify_dashboard_integrity.js # 34-point automated research audit
├── generate_dashboard_dataset.py     # Data enrichment & reconciliation script
├── package.json
├── tailwind.config.js
└── vite.config.js                    # Configured for host: true, port: 4173
```
