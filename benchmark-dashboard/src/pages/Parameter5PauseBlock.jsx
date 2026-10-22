import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid
} from 'recharts';
import { PauseCircle, AlertTriangle, ShieldCheck, HelpCircle, XCircle, Info, CheckCircle2 } from 'lucide-react';
import KpiCard from '../components/KpiCard';
import StatusBadge from '../components/StatusBadge';

export default function Parameter5PauseBlock({ clips, filteredMetrics, datasetFilter }) {
  const { blockMetrics, pauses } = filteredMetrics;

  const isBlockAnnotated = datasetFilter === 'all' || datasetFilter === 'UCLASS' || datasetFilter === 'FluencyBank';

  // Dynamic pause extraction from currently filtered clips
  const clipPauses = clips.flatMap(c => c.pauses || []);
  const totalPausesCount = clipPauses.length;
  const meanPauseSec = totalPausesCount > 0 ? (clipPauses.reduce((a, b) => a + b, 0) / totalPausesCount).toFixed(2) : '0.00';
  const pausesPerClip = (totalPausesCount / clips.length).toFixed(2);

  // Dynamic Pause duration histogram buckets
  const pauseDurationBuckets = [
    { range: '0.25 - 0.50s', count: clipPauses.filter(p => p >= 0.25 && p < 0.50).length },
    { range: '0.50 - 0.75s', count: clipPauses.filter(p => p >= 0.50 && p < 0.75).length },
    { range: '0.75 - 1.00s', count: clipPauses.filter(p => p >= 0.75 && p < 1.00).length },
    { range: '1.00 - 1.50s', count: clipPauses.filter(p => p >= 1.00 && p < 1.50).length },
    { range: '1.50 - 2.00s', count: clipPauses.filter(p => p >= 1.50 && p < 2.00).length },
    { range: '> 2.00s', count: clipPauses.filter(p => p >= 2.00).length },
  ];

  // Annotated Ground-Truth Block Clips (4 UCLASS, 2 FB)
  const allAnnotatedBlockClips = [
    { filename: 'M_1023_11y3m_1_dysfluent_027.wav', dataset: 'UCLASS', gtBlock: 1, predBlock: 0, predictedClass: 'Stutter (Prolongation)', outcome: 'False Negative' },
    { filename: 'M_1104_40y0m_1_dysfluent_037.wav', dataset: 'UCLASS', gtBlock: 1, predBlock: 0, predictedClass: 'Stutter (Repetition)', outcome: 'False Negative' },
    { filename: 'M_1104_40y0m_1_dysfluent_143.wav', dataset: 'UCLASS', gtBlock: 1, predBlock: 0, predictedClass: 'Filler (Interjection)', outcome: 'False Negative' },
    { filename: 'M_1104_40y0m_1_dysfluent_105.wav', dataset: 'UCLASS', gtBlock: 1, predBlock: 0, predictedClass: 'Filler (Hesitation)', outcome: 'False Negative' },
    { filename: 'FluencyBank_063_45.wav', dataset: 'FluencyBank', gtBlock: 1, predBlock: 0, predictedClass: 'Clean (No Burst)', outcome: 'False Negative' },
    { filename: 'FluencyBank_099_150.wav', dataset: 'FluencyBank', gtBlock: 1, predBlock: 0, predictedClass: 'Filler (Interjection)', outcome: 'False Negative' },
  ];

  const displayedBlockClips = datasetFilter === 'all'
    ? allAnnotatedBlockClips
    : allAnnotatedBlockClips.filter(c => c.dataset === datasetFilter);

  // False Positive Block Clips (4 in UCLASS)
  const allFpBlockClips = [
    { filename: 'M_1099_25y0m_1_dysfluent_059.wav', dataset: 'UCLASS', gtAnnotated: 'Sound Repetition', reason: 'Acoustic stop closure > 0.4s' },
    { filename: 'M_1105_21y0m_1_dysfluent_130.wav', dataset: 'UCLASS', gtAnnotated: 'Sound Repetition', reason: 'Acoustic stop closure > 0.4s' },
    { filename: 'M_1105_21y0m_1_dysfluent_239.wav', dataset: 'UCLASS', gtAnnotated: 'Interjection', reason: 'Pre-vocalic hesitation pause' },
    { filename: 'M_1105_21y0m_1_fluent_150.wav', dataset: 'UCLASS', gtAnnotated: 'Clean Fluent', reason: 'Plosive burst onset' },
  ];

  const displayedFpClips = datasetFilter === 'all'
    ? allFpBlockClips
    : allFpBlockClips.filter(c => c.dataset === datasetFilter);

  // Dataset-specific block stats computed dynamically
  const blockStats = React.useMemo(() => {
    const calc = (subClips, label) => {
      const annotated = subClips.filter(c => c.dataset !== 'Libri_Stutter' && c.gt_block !== null && c.gt_block !== undefined);
      if (annotated.length === 0) {
        return { tp: 'N/A', fp: 'N/A', fn: 'N/A', tn: 'N/A', total: subClips.length, spec: 'Not Evaluated', rec: 'Not Evaluated', f1: 'Not Evaluated', label: 'Annotation Unavailable' };
      }
      let tp = 0, fp = 0, fn = 0, tn = 0;
      annotated.forEach(c => {
        const gt = c.gt_block ? 1 : 0;
        const pred = c.pred_block ? 1 : 0;
        if (gt && pred) tp++;
        else if (!gt && pred) fp++;
        else if (gt && !pred) fn++;
        else tn++;
      });
      const spec = (tn + fp > 0) ? ((tn / (tn + fp)) * 100).toFixed(1) + '%' : '0.0%';
      const rec = (tp + fn > 0) ? ((tp / (tp + fn)) * 100).toFixed(1) + '%' : '0.0%';
      const prec = (tp + fp > 0) ? tp / (tp + fp) : 0;
      const recN = (tp + fn > 0) ? tp / (tp + fn) : 0;
      const f1 = (prec + recN > 0) ? ((2 * prec * recN) / (prec + recN)).toFixed(4) : '0.0000';
      return { tp, fp, fn, tn, total: annotated.length, spec, rec, f1, label };
    };

    const uclassClips = clips.filter(c => c.dataset === 'UCLASS');
    const fbClips = clips.filter(c => c.dataset === 'FluencyBank');
    const scopedClips = clips.filter(c => c.dataset === 'UCLASS' || c.dataset === 'FluencyBank');

    return {
      all: calc(scopedClips, `UCLASS + FB (Scoped N=${scopedClips.length})`),
      UCLASS: calc(uclassClips, `UCLASS (N=${uclassClips.length})`),
      FluencyBank: calc(fbClips, `FluencyBank (N=${fbClips.length})`),
      Libri_Stutter: { tp: 'N/A', fp: 'N/A', fn: 'N/A', tn: 'N/A', total: clips.filter(c => c.dataset === 'Libri_Stutter').length, spec: 'Not Evaluated', rec: 'Not Evaluated', f1: 'Not Evaluated', label: 'Annotation Unavailable' },
    };
  }, [clips]);

  const currentBlockStat = blockStats[datasetFilter] || blockStats.all;

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              Parameter 5: Pause & Articulatory Block Detection
            </h2>
            <StatusBadge 
              type={isBlockAnnotated ? "groundTruth" : "omitted"} 
              text={
                datasetFilter === 'all' 
                  ? `Scoped Block (UCLASS+FB N=${clips.filter(c => c.dataset !== 'Libri_Stutter').length}) • Acoustic Pauses (N=${clips.length})`
                  : datasetFilter === 'Libri_Stutter'
                  ? 'Pauses Evaluated • Blocks Unannotated'
                  : `${datasetFilter} Block Supervised (N=${clips.length})`
              } 
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Discriminating natural inter-word syntactic pauses from pre-phonation articulatory struggle blocks.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 p-2 rounded-lg border border-slate-200 text-slate-700">
          <span>Active Scope: <strong>{datasetFilter === 'all' ? `Combined (${clips.length} Clips)` : `${datasetFilter} (${clips.length} Clips)`}</strong></span>
          <span className="text-slate-300">•</span>
          <span>Pauses: <strong>{totalPausesCount}</strong></span>
          <span className="text-slate-300">•</span>
          <span>Mean Pause: <strong>{meanPauseSec}s</strong></span>
        </div>
      </div>

      {/* Forensic Reconciliation Box */}
      <div className={`rounded-xl p-4 text-xs space-y-2 border ${
        isBlockAnnotated ? 'bg-amber-50 border-amber-200 text-amber-950' : 'bg-slate-50 border-slate-200 text-slate-700'
      }`}>
        <div className="flex items-center gap-2 font-bold text-slate-900">
          {isBlockAnnotated ? (
            <HelpCircle className="w-4 h-4 text-amber-700 shrink-0" />
          ) : (
            <Info className="w-4 h-4 text-blue-600 shrink-0" />
          )}
          {isBlockAnnotated 
            ? 'Forensic Reconciliation: Why Block F1 = 0.0000 while Specificity is High' 
            : 'Libri_Stutter Annotation Status: Block Ground Truth Unavailable'}
        </div>
        <p className="leading-relaxed">
          {isBlockAnnotated ? (
            <>
              1. <strong>Co-Occurrence in 3s Slices:</strong> The acoustic block detector searches for an isolated pre-phonation silent hesitation (&ge; 0.40s) followed by an explosive burst (&ge; 2.5x). Across the 6 annotated clinical block clips (4 in UCLASS, 2 in FluencyBank), struggle co-occurred with audible repetitions or interjections without isolated silence. All 6 were classified as repetitions, prolongations, or fillers (<span className="font-semibold text-rose-800">Recall = 0.00%</span>).
              <br />
              2. <strong>Zero Silent Negative Assumption on Libri_Stutter:</strong> The Libri_Stutter annotation schema does not label blocks. Audiobook readers' stop closures before plosives (`/p, t, k, b/`) triggered the struggle-gap ratio in 19 clips. <em>Treating missing annotations as negative ground truth would fabricate 19 false alarms</em>. Scoping block evaluation strictly to corpora that annotate blocks confirms <strong>94.59% specificity</strong> on UCLASS+FB.
            </>
          ) : (
            <>
              <strong>Libri_Stutter</strong> consists of read audiobook recordings where phonetic annotations do not identify clinical blocks.
              <br />
              To uphold research integrity, block metrics are labeled <strong>Not evaluated — annotation unavailable</strong>. We strictly refuse to treat missing annotations as negative ground truth. Continuous acoustic pause statistics (total <strong>{totalPausesCount} pauses</strong>, mean <strong>{meanPauseSec}s</strong>) remain fully evaluated below.
            </>
          )}
        </p>
      </div>

      {/* Dynamic KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title={isBlockAnnotated ? "Block Specificity" : "Block Evaluation Status"}
          value={currentBlockStat.spec}
          subtitle={
            isBlockAnnotated 
              ? `${currentBlockStat.tn} TN of ${typeof currentBlockStat.tn === 'number' ? currentBlockStat.tn + currentBlockStat.fp : 0} controls`
              : "Annotation unavailable in schema"
          }
          delta={currentBlockStat.label}
          deltaType={isBlockAnnotated ? "positive" : "negative"}
          badgeType={isBlockAnnotated ? "groundTruth" : "omitted"}
          tooltip="True negative rate on corpora with verified clinical block annotations."
          icon={ShieldCheck}
        />
        <KpiCard
          title="Block Sensitivity (Recall)"
          value={currentBlockStat.rec}
          subtitle={
            isBlockAnnotated 
              ? `${currentBlockStat.tp} TP of ${typeof currentBlockStat.fn === 'number' ? currentBlockStat.tp + currentBlockStat.fn : 0} annotated blocks`
              : "Unannotated dataset"
          }
          delta={isBlockAnnotated ? `FN: ${currentBlockStat.fn} (Co-occurring)` : "Schema Omission"}
          deltaType={isBlockAnnotated ? "negative" : "neutral"}
          badgeType={isBlockAnnotated ? "warning" : "omitted"}
          tooltip="Zero observed sensitivity because clinical blocks co-occurred with repetitions/fillers in 3s windows."
          icon={XCircle}
        />
        <KpiCard
          title="Syntactic Pauses Detected"
          value={totalPausesCount}
          subtitle={`Mean ${pausesPerClip} pauses per clip`}
          delta={`${datasetFilter === 'all' ? `All ${clips.length} Clips` : `${datasetFilter} (${clips.length} Clips)`}`}
          deltaType="neutral"
          badgeType="proxy"
          tooltip="Acoustic pauses (gap >= 0.25s) extracted from ASR word timestamps."
          icon={PauseCircle}
        />
        <KpiCard
          title="Mean Pause Duration"
          value={`${meanPauseSec} sec`}
          subtitle="Inter-word silence duration"
          delta="Dynamic Calculation"
          deltaType="neutral"
          badgeType="proxy"
          tooltip="Average duration of inter-word silence across the selected clips."
          icon={PauseCircle}
        />
      </div>

      {/* SECTION 1: CLINICAL BLOCK DETECTION */}
      {isBlockAnnotated ? (
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-base font-semibold text-slate-900">
                Section A: Clinical Block Struggle Detection ({currentBlockStat.label})
              </h3>
              <p className="text-xs text-slate-500">
                Evaluated on corpora with clinical block ground truth ({currentBlockStat.label})
              </p>
            </div>
            <StatusBadge type="groundTruth" text={`Specificity: ${currentBlockStat.spec}`} />
          </div>

          {/* 2x2 Contingency Table for Blocks */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Block Contingency Matrix ({currentBlockStat.label})
              </h4>
              <div className="border border-slate-200 rounded-lg overflow-hidden text-xs">
                <table className="w-full text-center">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="p-2.5 text-left">Ground Truth</th>
                      <th className="p-2.5 bg-blue-50/50">Pred Block</th>
                      <th className="p-2.5 bg-slate-50">Pred No Block</th>
                      <th className="p-2.5 font-bold">Total</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    <tr>
                      <td className="p-2.5 text-left font-sans font-medium text-slate-900">Actual Block</td>
                      <td className="p-2.5 font-bold text-emerald-700 bg-emerald-50/30">TP = {currentBlockStat.tp}</td>
                      <td className="p-2.5 font-bold text-rose-600 bg-rose-50/30">FN = {currentBlockStat.fn}</td>
                      <td className="p-2.5 font-semibold text-slate-800">{currentBlockStat.tp + currentBlockStat.fn}</td>
                    </tr>
                    <tr>
                      <td className="p-2.5 text-left font-sans font-medium text-slate-900">Actual Non-Block</td>
                      <td className="p-2.5 font-bold text-amber-600 bg-amber-50/30">FP = {currentBlockStat.fp}</td>
                      <td className="p-2.5 font-bold text-emerald-700 bg-emerald-50/30">TN = {currentBlockStat.tn}</td>
                      <td className="p-2.5 font-semibold text-slate-800">{currentBlockStat.fp + currentBlockStat.tn}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Block Diagnostic Metrics
              </h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <div className="text-slate-500 text-[11px]">Precision</div>
                  <div className="text-sm font-bold text-slate-900">0.00%</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <div className="text-slate-500 text-[11px]">Recall / Sensitivity</div>
                  <div className="text-sm font-bold text-rose-600">{currentBlockStat.rec}</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <div className="text-slate-500 text-[11px]">Specificity</div>
                  <div className="text-sm font-bold text-emerald-600">{currentBlockStat.spec}</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <div className="text-slate-500 text-[11px]">F1 Score</div>
                  <div className="text-sm font-bold text-slate-700">{currentBlockStat.f1}</div>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 italic leading-relaxed">
                High specificity reflects low false alarm rate on non-block controls. Sensitivity was zero because clinical blocks co-occurred with repetitions/prolongations.
              </p>
            </div>
          </div>

          {/* Audit of annotated block clips */}
          {displayedBlockClips.length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-100">
              <h4 className="text-xs font-semibold text-slate-800 mb-2">
                Audit Table: Ground-Truth Annotated Block Clips ({displayedBlockClips.length} Clips)
              </h4>
              <div className="overflow-x-auto border border-slate-200 rounded-lg text-xs">
                <table className="w-full text-left border-collapse">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="p-2.5">Clip Filename</th>
                      <th className="p-2.5">Dataset</th>
                      <th className="p-2.5">Predicted Dominant Subtype</th>
                      <th className="p-2.5">Outcome</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                    {displayedBlockClips.map((c, i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        <td className="p-2.5 text-slate-800 font-medium">{c.filename}</td>
                        <td className="p-2.5 text-slate-600">{c.dataset}</td>
                        <td className="p-2.5 text-slate-700">{c.predictedClass}</td>
                        <td className="p-2.5 text-rose-600 font-semibold">{c.outcome}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 text-center shadow-sm space-y-2">
          <AlertTriangle className="w-8 h-8 text-amber-500 mx-auto" />
          <h3 className="text-base font-bold text-slate-900">
            Clinical Block Ground Truth Unavailable for Libri_Stutter
          </h3>
          <p className="text-xs text-slate-500 max-w-xl mx-auto leading-relaxed">
            The Libri_Stutter corpus does not include clinical block struggle annotations in its ground truth schema. To maintain scientific integrity, block metrics are omitted rather than assuming all clips are negative controls.
          </p>
        </div>
      )}

      {/* SECTION 2: CONTINUOUS PAUSE SIGNAL ANALYSIS */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Section B: Continuous Acoustic Pause Signal Analysis ({datasetFilter === 'all' ? `All ${clips.length} Clips` : `${datasetFilter} (${clips.length} Clips)`})
            </h3>
            <p className="text-xs text-slate-500">
              {totalPausesCount} silence events (&ge; 0.25s) • Mean Duration: {meanPauseSec}s
            </p>
          </div>
          <StatusBadge type="proxy" text="Acoustic Signal Processing" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Pause Duration Histogram */}
          <div className="lg:col-span-8">
            <h4 className="text-xs font-semibold text-slate-700 mb-2">
              Pause Duration Distribution Histogram ({totalPausesCount} Pauses)
            </h4>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={pauseDurationBuckets} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="range" tick={{ fontSize: 10 }} stroke="#64748b" />
                  <YAxis tick={{ fontSize: 11 }} stroke="#64748b" allowDecimals={false} />
                  <Tooltip
                    formatter={(val) => [`${val} pauses`]}
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  />
                  <Bar dataKey="count" name="Pause Events" fill="#0d9488" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Pause Summary Box */}
          <div className="lg:col-span-4 bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3 text-xs">
            <h4 className="font-semibold text-slate-900">
              {datasetFilter === 'all' ? 'Combined' : datasetFilter} Pause Telemetry
            </h4>
            <div className="space-y-2">
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-600">Total Silence Events:</span>
                <span className="font-bold text-slate-900">{totalPausesCount}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-600">Mean Duration:</span>
                <span className="font-bold text-teal-700">{meanPauseSec} sec</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-600">Pauses Per Clip:</span>
                <span className="font-bold text-slate-900">{pausesPerClip}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-600">Silence Threshold:</span>
                <span className="font-mono text-slate-700">&ge; 0.25 sec</span>
              </div>
            </div>
            <p className="text-[11px] text-slate-500 italic pt-2 border-t border-slate-200">
              Pauses are extracted deterministically from ASR word boundaries and energy silence segmentations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
