import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, PieChart, Pie, Cell
} from 'recharts';
import { CheckCircle2, AlertTriangle, ShieldCheck, HelpCircle } from 'lucide-react';
import KpiCard from '../components/KpiCard';
import StatusBadge from '../components/StatusBadge';
import { formatPercent, formatNumber } from '../utils/metrics';

export default function Parameter3CleanSpeech({ clips, filteredMetrics, datasetFilter }) {
  const { multiclass } = filteredMetrics;
  const cleanClass = multiclass.perClass.Clean;

  // Per-Dataset Clean Metrics computed dynamically
  const datasetCleanStats = React.useMemo(() => {
    return ['UCLASS', 'FluencyBank', 'Libri_Stutter'].map(dName => {
      const dClips = clips.filter(c => c.dataset === dName);
      const total = dClips.length;
      let tp = 0, fp = 0, fn = 0, tn = 0, gtClean = 0, predClean = 0;
      dClips.forEach(c => {
        const isGt = c.gt_dominant === 'Clean';
        const isPred = c.pred_dominant === 'Clean';
        if (isGt) gtClean++;
        if (isPred) predClean++;
        if (isGt && isPred) tp++;
        else if (!isGt && isPred) fp++;
        else if (isGt && !isPred) fn++;
        else tn++;
      });
      const recall = gtClean > 0 ? Number(((tp / gtClean) * 100).toFixed(1)) : 0;
      const specificity = (tn + fp) > 0 ? Number(((tn / (tn + fp)) * 100).toFixed(1)) : 0;
      return { dataset: dName, total, gtClean, predClean, tp, fp, recall, specificity };
    });
  }, [clips]);

  // Error examples
  const falseCleanClips = clips.filter(c => c.gt_dominant !== 'Clean' && c.pred_dominant === 'Clean');
  const missedCleanClips = clips.filter(c => c.gt_dominant === 'Clean' && c.pred_dominant !== 'Clean');

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              Parameter 3: Clean-Speech & Fluency Recognition
            </h2>
            <StatusBadge type="groundTruth" text={`Supervised Fluent Holdout Evaluation (N=${clips.length})`} />
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Correct recognition of fluent speech without false alarms across {cleanClass.support} clinically verified non-stutter speech segments.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 p-2 rounded-lg border border-slate-200 text-slate-700">
          <span>Specificity: <strong>{formatPercent(cleanClass.specificity)}</strong></span>
          <span className="text-slate-300">•</span>
          <span>Precision: <strong>{formatPercent(cleanClass.precision)}</strong></span>
          <span className="text-slate-300">•</span>
          <span>Recall: <strong>{formatPercent(cleanClass.recall)}</strong></span>
        </div>
      </div>

      {/* Scope Disclaimer */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-700 flex items-start gap-3">
        <HelpCircle className="w-5 h-5 text-slate-500 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-semibold text-slate-900">
            Task Definition & Clinical Scope Note
          </div>
          <p className="leading-relaxed">
            The clean class represents a <strong>mutually exclusive dominant label</strong> in the 3-class classification framework (Clean vs. Stutter vs. Filler). A high specificity of <strong>{formatPercent(cleanClass.specificity)}</strong> confirms that SpeechAI suppresses false dysfluency alarms on most fluent intervals. However, this screening-level classification is not an exhaustive clinical evaluation of communicative fluency or naturalness.
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Fluent Specificity"
          value={formatPercent(cleanClass.specificity)}
          subtitle={`${cleanClass.tn} TN out of ${cleanClass.tn + cleanClass.fp} non-clean`}
          delta={`${cleanClass.tn} / ${cleanClass.tn + cleanClass.fp} Rejected`}
          deltaType="positive"
          badgeType="groundTruth"
          tooltip="True negative rate: actual dysfluent clips correctly avoided by the clean classifier."
          icon={ShieldCheck}
        />
        <KpiCard
          title="Clean Precision"
          value={formatPercent(cleanClass.precision)}
          subtitle={`${cleanClass.tp} TP out of ${cleanClass.tp + cleanClass.fp} predicted clean`}
          delta={`${cleanClass.tp} / ${cleanClass.tp + cleanClass.fp} Correct`}
          deltaType="neutral"
          badgeType="groundTruth"
          tooltip={`When the model predicts clean, it is verified correct ${formatPercent(cleanClass.precision)} of the time.`}
          icon={CheckCircle2}
        />
        <KpiCard
          title="Clean Recall"
          value={formatPercent(cleanClass.recall)}
          subtitle={`${cleanClass.tp} TP out of ${cleanClass.support} ground truth clean`}
          delta={`${cleanClass.tp} / ${cleanClass.support} Captured`}
          deltaType="neutral"
          badgeType="groundTruth"
          tooltip="Sensitivity on fluent intervals. Many clean clips were tagged with subtle fillers or pauses."
          icon={AlertTriangle}
        />
        <KpiCard
          title="Clean Class F1"
          value={formatNumber(cleanClass.f1, 4)}
          subtitle="Harmonic mean of clean P & R"
          delta="Benchmark Score"
          deltaType="neutral"
          badgeType="groundTruth"
          tooltip={`Multiclass F1 for the clean speech class across all ${clips.length} validation clips.`}
          icon={CheckCircle2}
        />
      </div>

      {/* Row 2: Clean Contingency Table & Error Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <h3 className="text-base font-semibold text-slate-900 mb-1">
            Clean-Speech Contingency Table (N={clips.length})
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Actual Fluent Ground Truth vs. Predicted Clean Classification
          </p>

          <div className="grid grid-cols-2 gap-3 text-center">
            <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200">
              <span className="text-[11px] font-semibold text-emerald-800 uppercase block tracking-wider">
                True Positives (TP)
              </span>
              <span className="text-3xl font-extrabold text-emerald-700 block mt-1">
                {cleanClass.tp}
              </span>
              <span className="text-[11px] text-emerald-600 mt-0.5 block">
                Correctly Recognized Fluent
              </span>
            </div>

            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200">
              <span className="text-[11px] font-semibold text-rose-800 uppercase block tracking-wider">
                False Clean (FP)
              </span>
              <span className="text-3xl font-extrabold text-rose-700 block mt-1">
                {cleanClass.fp}
              </span>
              <span className="text-[11px] text-rose-600 mt-0.5 block">
                Dysfluent Mispredicted as Clean
              </span>
            </div>

            <div className="p-4 rounded-xl bg-amber-50 border border-amber-200">
              <span className="text-[11px] font-semibold text-amber-800 uppercase block tracking-wider">
                Missed Clean (FN)
              </span>
              <span className="text-3xl font-extrabold text-amber-700 block mt-1">
                {cleanClass.fn}
              </span>
              <span className="text-[11px] text-amber-600 mt-0.5 block">
                Fluent Flagged as Dysfluent
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
              <span className="text-[11px] font-semibold text-slate-700 uppercase block tracking-wider">
                True Negatives (TN)
              </span>
              <span className="text-3xl font-extrabold text-slate-800 block mt-1">
                {cleanClass.tn}
              </span>
              <span className="text-[11px] text-slate-500 mt-0.5 block">
                Dysfluencies Correctly Excluded
              </span>
            </div>
          </div>
        </div>

        {/* Per-Dataset Clean Performance */}
        <div className="lg:col-span-6 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <h3 className="text-base font-semibold text-slate-900 mb-1">
            Clean-Speech Specificity by Dataset
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Evaluating true negative rejection rates across clinical and synthetic audio
          </p>

          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={datasetCleanStats} margin={{ top: 15, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="dataset" tick={{ fontSize: 11 }} stroke="#64748b" />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} stroke="#64748b" unit="%" />
                <Tooltip
                  formatter={(val) => [`${val}%`]}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Bar dataKey="specificity" name="Clean Specificity (True Negative %)" fill="#10b981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="recall" name="Clean Recall (Sensitivity %)" fill="#0284c7" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Row 3: Inspect Error Cases */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* False Clean (Under-detection) */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-rose-900">
              False Clean Cases (Dysfluencies Missed as Clean: {falseCleanClips.length})
            </h4>
            <span className="text-[10px] px-2 py-0.5 rounded bg-rose-50 text-rose-700 font-medium">Under-detection</span>
          </div>
          <div className="space-y-1.5 max-h-52 overflow-y-auto text-xs">
            {falseCleanClips.slice(0, 8).map((c, idx) => (
              <div key={idx} className="p-2 rounded bg-slate-50 border border-slate-100 flex items-center justify-between">
                <div>
                  <span className="font-mono text-slate-700">{c.filename}</span>
                  <span className="text-[11px] text-slate-400 block">{c.dataset} • Actual: <strong>{c.gt_dominant}</strong></span>
                </div>
                <span className="text-[11px] font-semibold text-rose-600 bg-rose-50 px-2 py-0.5 rounded">
                  Pred: Clean
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Missed Clean (Over-detection) */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-amber-900">
              Missed Clean Cases (Fluent Flagged as Dysfluent: {missedCleanClips.length})
            </h4>
            <span className="text-[10px] px-2 py-0.5 rounded bg-amber-50 text-amber-700 font-medium">Over-detection</span>
          </div>
          <div className="space-y-1.5 max-h-52 overflow-y-auto text-xs">
            {missedCleanClips.slice(0, 8).map((c, idx) => (
              <div key={idx} className="p-2 rounded bg-slate-50 border border-slate-100 flex items-center justify-between">
                <div>
                  <span className="font-mono text-slate-700">{c.filename}</span>
                  <span className="text-[11px] text-slate-400 block">{c.dataset} • Actual: <strong>Clean</strong></span>
                </div>
                <span className="text-[11px] font-semibold text-amber-600 bg-amber-50 px-2 py-0.5 rounded">
                  Pred: {c.pred_dominant}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
