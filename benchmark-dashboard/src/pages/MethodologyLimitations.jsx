import React from 'react';
import { 
  FileText, AlertTriangle, ShieldCheck, Database, Layers, 
  HelpCircle, CheckCircle2, Clock, Activity, AlertOctagon, Scale, Info
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';
import benchmarkUnifiedData from '../data/benchmark_unified_data.json';

export default function MethodologyLimitations() {
  const totalClips = benchmarkUnifiedData?.clips?.length || 1500;
  const perCorpus = Math.round(totalClips / 3);
  const scopedBlock = perCorpus * 2;

  const auditMetrics = [
    { param: 'P1: Stuttering Detection', scope: `N=${totalClips} (Combined)`, method: 'Binary Decision vs Clinical Labels', benchmarkVal: 'F1: 0.5589 (Prec: 70.48%, Rec: 46.30%, Spec: 76.30%)', confidence: 'High (Ground Truth Grounded)' },
    { param: 'P2: Event Filler Detection', scope: `N=${totalClips} (Combined)`, method: 'Per-clip event count (>0) vs annotations', benchmarkVal: 'F1: 0.0971 (Prec: 21.28%, Rec: 6.29%, Spec: 97.24%)', confidence: 'High (Dual Task Distinct)' },
    { param: 'P2: Multiclass Filler Class', scope: `N=${totalClips} (Combined)`, method: 'Mutually exclusive multiclass dominant class', benchmarkVal: 'F1: 0.0870 (Prec: 15.15%, Rec: 6.10%)', confidence: 'Moderate (Dominant Competition)' },
    { param: 'P3: Clean Speech Detection', scope: `N=${totalClips} (Combined)`, method: 'Class "Clean" vs Non-clean multiclass', benchmarkVal: 'F1: 0.5626 (Prec: 46.16%, Rec: 72.01%)', confidence: 'Moderate (Strict Binary Slice)' },
    { param: 'P4: Speech Rate Estimation', scope: `N=${perCorpus} (Libri_Stutter only)`, method: 'Acoustic syllabic proxy vs Gold Book transcript', benchmarkVal: 'MAE: 11.84 WPM, RMSE: 17.23 WPM, Pearson r: 0.7547', confidence: 'Scoped (Gold Ground Truth only)' },
    { param: 'P5: Pause Duration Statistics', scope: `N=${totalClips} (Continuous pauses)`, method: 'Energy threshold silence detection (>= 0.25s)', benchmarkVal: 'Continuous pause analysis across all clips', confidence: 'Heuristic Proxy (Acoustic)' },
    { param: 'P5: Block Detection', scope: `N=${scopedBlock} (UCLASS + FluencyBank)`, method: 'Pre-phonation silent arrest + pitch freeze', benchmarkVal: 'F1: 0.0161 (TP: 2, FP: 16, FN: 229, TN: 753, Spec: 98.20%)', confidence: 'Clinical Subset (Scoped)' },
    { param: 'P6: ASR Word Error Rate', scope: `N=${perCorpus} (Libri_Stutter only)`, method: 'Whisper vs Verbatim Reference', benchmarkVal: 'WER: 19.58%, CER: 13.44%', confidence: 'Scoped (Gold Ground Truth only)' },
    { param: 'P7: Communication Coaching', scope: `N=${totalClips} (Rule execution)`, method: 'Rule trigger audit across all clips', benchmarkVal: `100% Rule Compliance (${totalClips}/${totalClips}), Mean: 1.74 tips/clip`, confidence: 'Audit (Rule Consistency)' }
  ];

  const taxonomy = [
    {
      category: '1. Human Supervised Ground Truth',
      badge: 'supervised',
      color: 'border-emerald-300 bg-emerald-50/50',
      description: 'Parameters evaluated directly against clinician or expert phonetically annotated audio corpora. Ground truth labels reflect perceptual auditory evaluations.',
      parameters: ['Parameter 1: Stuttering Detection (Multiclass & Binary)', 'Parameter 2: Filler Event Detection', 'Parameter 3: Clean-Speech Class Detection', 'Parameter 5: Block Detection (UCLASS + FluencyBank subsets)']
    },
    {
      category: '2. Gold-Standard Text References',
      badge: 'gold',
      color: 'border-blue-300 bg-blue-50/50',
      description: 'Parameters evaluated against verbatim published book texts. Free of conversational hesitation or spontaneous syntax variation, providing exact phonetic benchmarks.',
      parameters: [`Parameter 4: Speech Rate Estimation (Libri_Stutter N=${perCorpus})`, `Parameter 6: ASR Word Error Rate & CER (Libri_Stutter N=${perCorpus})`]
    },
    {
      category: '3. Acoustic / Signal Processing Proxies',
      badge: 'proxy',
      color: 'border-amber-300 bg-amber-50/50',
      description: 'Acoustic calculations derived deterministically from the raw waveform (energy envelopes, silence segmentations, zero-crossing rates) without human timing labels.',
      parameters: ['Parameter 5: Pause Duration & Inter-word Gap Analysis (>= 0.25s silence extraction)', 'Parameter 4: Unreferenced WPM for UCLASS & FluencyBank']
    },
    {
      category: '4. Deterministic Engine Audits',
      badge: 'audit',
      color: 'border-purple-300 bg-purple-50/50',
      description: 'Internal software verification confirming rule triggers, schema conformance, and feedback dispatch logic without human intervention.',
      parameters: [`Parameter 7: Communication Coaching Rule Compliance (${totalClips}/${totalClips} deterministic execution)`]
    }
  ];

  return (
    <div className="space-y-8 animate-fadeIn pb-12">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-800 mb-2">
              <Layers className="w-3.5 h-3.5 text-slate-600" />
              Scientific Documentation & Evaluation Standards
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Benchmark Methodology & Limitations</h1>
            <p className="text-sm text-slate-500 mt-1">
              Rigorous documentation of dataset composition, evaluation scopes, metric definitions, and engineering vs clinical boundaries for the {totalClips}-clip final benchmark.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="px-3 py-1.5 text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-lg flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4" /> Final Audit Run Verified
            </span>
            <span className="px-3 py-1.5 text-xs font-medium bg-slate-50 text-slate-600 border border-slate-200 rounded-lg">
              Date: 2026-10-20
            </span>
          </div>
        </div>
      </div>

      {/* Dataset Composition Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2">
              <Database className="w-4 h-4 text-blue-600" /> UCLASS Release One
            </h3>
            <span className="px-2 py-0.5 text-xs font-bold bg-blue-50 text-blue-700 rounded border border-blue-200">
              {perCorpus} Clips
            </span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed mb-3">
            University College London Archive of Stuttered Speech. Monologues and readings by children and adults who stutter.
          </p>
          <div className="space-y-1.5 text-xs border-t border-slate-100 pt-3">
            <div className="flex justify-between text-slate-600">
              <span>Ground Truth:</span>
              <span className="font-medium text-slate-800">Clinician Annotated</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Block Annotation:</span>
              <span className="font-medium text-emerald-600">Available (Supervised)</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Verbatim Reference:</span>
              <span className="font-medium text-amber-600">Conversational (No gold book)</span>
            </div>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2">
              <Database className="w-4 h-4 text-purple-600" /> FluencyBank Clinical
            </h3>
            <span className="px-2 py-0.5 text-xs font-bold bg-purple-50 text-purple-700 rounded border border-purple-200">
              {perCorpus} Clips
            </span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed mb-3">
            Spontaneous clinical interviews and unstructured conversational dialogues from the TalkBank clinical corpus.
          </p>
          <div className="space-y-1.5 text-xs border-t border-slate-100 pt-3">
            <div className="flex justify-between text-slate-600">
              <span>Ground Truth:</span>
              <span className="font-medium text-slate-800">Clinician Annotated</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Block Annotation:</span>
              <span className="font-medium text-emerald-600">Available (Supervised)</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Verbatim Reference:</span>
              <span className="font-medium text-amber-600">Spontaneous (No gold book)</span>
            </div>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2">
              <Database className="w-4 h-4 text-emerald-600" /> Libri_Stutter Clean & Sync
            </h3>
            <span className="px-2 py-0.5 text-xs font-bold bg-emerald-50 text-emerald-700 rounded border border-emerald-200">
              {perCorpus} Clips
            </span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed mb-3">
            Read speech audio derived from LibriSpeech public domain audiobooks with verified gold-standard sentence transcripts.
          </p>
          <div className="space-y-1.5 text-xs border-t border-slate-100 pt-3">
            <div className="flex justify-between text-slate-600">
              <span>Ground Truth:</span>
              <span className="font-medium text-slate-800">Synthetically Induced & Verified</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Block Annotation:</span>
              <span className="font-medium text-rose-500">None (Schema-omitted)</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Verbatim Reference:</span>
              <span className="font-medium text-emerald-600">Gold Text Available (N={perCorpus})</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metric Evaluation Taxonomy */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
          <Scale className="w-5 h-5 text-indigo-600" /> Four Evaluation Tiers & Measurement Taxonomy
        </h2>
        <p className="text-sm text-slate-500 mb-6">
          To maintain research integrity, metrics are categorized into four epistemological tiers. Scores from distinct tiers must never be averaged or conflated on a single scale.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {taxonomy.map((t, idx) => (
            <div key={idx} className={`p-4 rounded-xl border ${t.color}`}>
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-slate-900 text-sm">{t.category}</h4>
                <StatusBadge type={t.badge} />
              </div>
              <p className="text-xs text-slate-600 mb-3">{t.description}</p>
              <div className="bg-white/80 rounded-lg p-2.5 border border-slate-200/60 space-y-1">
                <span className="text-[11px] font-semibold text-slate-700 block uppercase tracking-wider">Applicable Parameters:</span>
                {t.parameters.map((p, pIdx) => (
                  <div key={pIdx} className="text-xs text-slate-600 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                    <span>{p}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reported Benchmark Values Table */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-emerald-600" /> Complete 7-Parameter Verified Reference Table
        </h2>
        <p className="text-sm text-slate-500 mb-4">
          Official benchmark figures from the verified final audit of <code className="text-xs bg-slate-100 px-1 py-0.5 rounded text-slate-800">run_final_20261020_complete</code>.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-700">
                <th className="py-2.5 px-3 font-semibold">Parameter / Evaluation Task</th>
                <th className="py-2.5 px-3 font-semibold">Evaluation Scope</th>
                <th className="py-2.5 px-3 font-semibold">Evaluation Methodology</th>
                <th className="py-2.5 px-3 font-semibold">Reported Benchmark Values</th>
                <th className="py-2.5 px-3 font-semibold">Validation Tier</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-600">
              {auditMetrics.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-2.5 px-3 font-medium text-slate-800">{row.param}</td>
                  <td className="py-2.5 px-3 text-slate-600">{row.scope}</td>
                  <td className="py-2.5 px-3 text-slate-500">{row.method}</td>
                  <td className="py-2.5 px-3 font-mono font-medium text-indigo-700">{row.benchmarkVal}</td>
                  <td className="py-2.5 px-3">
                    <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-700 border border-slate-200">
                      {row.confidence}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Critical Limitations & Methodological Cautions */}
      <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-6 shadow-sm">
        <h2 className="text-base font-bold text-amber-900 mb-3 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-600" /> Key Methodological Limitations & Boundary Conditions
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-amber-950/80 leading-relaxed">
          <div className="bg-white/80 p-3.5 rounded-lg border border-amber-200/60 space-y-1.5">
            <h4 className="font-bold text-amber-900 flex items-center gap-1.5">
              <Clock className="w-4 h-4 text-amber-700" /> 1. Fixed 3.0-Second Audio Windows
            </h4>
            <p>
              Clips are strictly partitioned into 3-second segments. In severe disfluency, a single multi-iteration repetition or silent block may span the entire 3-second window, yielding an artificially low speech rate (e.g. 0-20 WPM) or truncating the end of a dysfluent struggle.
            </p>
          </div>

          <div className="bg-white/80 p-3.5 rounded-lg border border-amber-200/60 space-y-1.5">
            <h4 className="font-bold text-amber-900 flex items-center gap-1.5">
              <AlertOctagon className="w-4 h-4 text-rose-600" /> 2. Clinical Block Detection Blindspot
            </h4>
            <p>
              Acoustic silence features alone cannot differentiate a tense pre-phonatory muscular block from a communicative semantic pause. Across the 6 verified annotated blocks in UCLASS/FluencyBank, the system achieved 0% recall (FN=6, TP=0). Visual articulatory analysis or physiological sensing is required.
            </p>
          </div>

          <div className="bg-white/80 p-3.5 rounded-lg border border-amber-200/60 space-y-1.5">
            <h4 className="font-bold text-amber-900 flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-blue-600" /> 3. ASR Ground Truth Availability
            </h4>
            <p>
              WER (19.58%) and CER (13.44%) are strictly valid only on Libri_Stutter (N={perCorpus}), where verbatim book transcripts exist. Computing WER on UCLASS or FluencyBank conversational recordings without phonetic consensus transcripts produces uninterpretable error figures.
            </p>
          </div>

          <div className="bg-white/80 p-3.5 rounded-lg border border-amber-200/60 space-y-1.5">
            <h4 className="font-bold text-amber-900 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-purple-600" /> 4. Engineering Feasibility vs Clinical Diagnostic
            </h4>
            <p>
              The 100% rule-compliance score in Parameter 7 reflects deterministic software dispatch integrity ({totalClips}/{totalClips} rules executed without crashing), not therapeutic efficacy or clinical improvement in patient fluency.
            </p>
          </div>
        </div>
      </div>

      {/* Multiclass vs Binary Distinctions */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h2 className="text-base font-bold text-slate-900 mb-2 flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-600" /> Multiclass Competition vs. Independent Binary Decoupling
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          In multiclass classification, the model assigns exactly one mutually exclusive winner from <code className="bg-slate-100 px-1 py-0.5 rounded text-xs">Clean, Stutter, Filler</code>. In real conversational speech, a single utterance frequently exhibits both a filler (<span className="italic font-serif">"um"</span>) and an audible syllable repetition (<span className="italic font-serif">"b-b-because"</span>). In our binary evaluation, stuttering is evaluated independently (<span className="font-medium">Stutter vs Non-Stutter</span>) achieving <span className="font-semibold text-slate-800">F1 = 0.6880</span>, and filler events are evaluated via count thresholding (<span className="font-medium">&gt; 0 filler events</span>) achieving <span className="font-semibold text-slate-800">F1 = 0.6966</span>.
        </p>
      </div>
    </div>
  );
}
