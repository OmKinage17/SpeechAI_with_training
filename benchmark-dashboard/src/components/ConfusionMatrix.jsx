import React from 'react';

export default function ConfusionMatrix({
  matrix,
  labels = ['Clean', 'Stutter', 'Filler'],
  title = 'Confusion Matrix',
  subtitle = 'Actual Ground Truth (Rows) vs. Predicted Class (Columns)'
}) {
  if (!matrix) return null;

  // Calculate row totals (Support) and column totals
  const rowTotals = {};
  const colTotals = {};
  labels.forEach(l => {
    rowTotals[l] = 0;
    colTotals[l] = 0;
  });

  labels.forEach(actual => {
    labels.forEach(pred => {
      const val = matrix[actual] ? matrix[actual][pred] || 0 : 0;
      rowTotals[actual] += val;
      colTotals[pred] += val;
    });
  });

  const totalClips = Object.values(rowTotals).reduce((a, b) => a + b, 0);

  // Helper for cell heat coloring
  const getCellBg = (actual, pred, val) => {
    if (val === 0) return 'bg-slate-50 text-slate-400';
    if (actual === pred) {
      // True positive diagonal - Emerald gradient
      const ratio = rowTotals[actual] > 0 ? val / rowTotals[actual] : 0;
      if (ratio > 0.7) return 'bg-emerald-600 text-white font-bold shadow-sm';
      if (ratio > 0.4) return 'bg-emerald-500 text-white font-semibold';
      return 'bg-emerald-100 text-emerald-900 font-medium';
    } else {
      // Off-diagonal error - Amber/Rose subtle
      const ratio = rowTotals[actual] > 0 ? val / rowTotals[actual] : 0;
      if (ratio > 0.3) return 'bg-rose-100 text-rose-900 font-semibold';
      return 'bg-amber-50 text-amber-900';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        <p className="text-xs text-slate-500">{subtitle}</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-center border-collapse text-xs">
          <thead>
            <tr>
              <th className="p-2 border-b border-r border-slate-200 bg-slate-50 text-slate-400 font-normal">
                Actual \ Pred
              </th>
              {labels.map(pred => (
                <th key={pred} className="p-2 border-b border-slate-200 bg-slate-50 font-semibold text-slate-700">
                  {pred}
                </th>
              ))}
              <th className="p-2 border-b border-l border-slate-200 bg-slate-100 font-semibold text-slate-700">
                Support
              </th>
              <th className="p-2 border-b border-l border-slate-200 bg-slate-100 font-semibold text-slate-700">
                Recall
              </th>
            </tr>
          </thead>
          <tbody>
            {labels.map(actual => {
              const tpVal = matrix[actual] ? matrix[actual][actual] || 0 : 0;
              const support = rowTotals[actual] || 0;
              const recall = support > 0 ? (tpVal / support) * 100 : 0;

              return (
                <tr key={actual} className="border-b border-slate-100">
                  <td className="p-2 border-r border-slate-200 bg-slate-50 font-semibold text-slate-800 text-left pl-3">
                    {actual}
                  </td>
                  {labels.map(pred => {
                    const val = matrix[actual] ? matrix[actual][pred] || 0 : 0;
                    return (
                      <td key={pred} className={`p-3 transition-colors ${getCellBg(actual, pred, val)}`}>
                        <div className="text-sm font-semibold">{val}</div>
                        {support > 0 && (
                          <div className="text-[10px] opacity-80">
                            {((val / support) * 100).toFixed(0)}%
                          </div>
                        )}
                      </td>
                    );
                  })}
                  <td className="p-2 border-l border-slate-200 bg-slate-50 font-medium text-slate-700">
                    {support}
                  </td>
                  <td className="p-2 border-l border-slate-200 bg-slate-50 font-semibold text-slate-900">
                    {recall.toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="border-t border-slate-200 bg-slate-100/70">
              <td className="p-2 border-r border-slate-200 font-semibold text-slate-800 text-left pl-3">
                Total Pred
              </td>
              {labels.map(pred => (
                <td key={pred} className="p-2 font-medium text-slate-700">
                  {colTotals[pred]}
                </td>
              ))}
              <td className="p-2 border-l border-slate-200 font-bold text-slate-900">
                {totalClips}
              </td>
              <td className="p-2 border-l border-slate-200 text-slate-400">
                —
              </td>
            </tr>
            <tr className="bg-slate-100/70">
              <td className="p-2 border-r border-slate-200 font-semibold text-slate-800 text-left pl-3">
                Precision
              </td>
              {labels.map(pred => {
                const tpVal = matrix[pred] ? matrix[pred][pred] || 0 : 0;
                const totalP = colTotals[pred] || 0;
                const prec = totalP > 0 ? (tpVal / totalP) * 100 : 0;
                return (
                  <td key={pred} className="p-2 font-semibold text-slate-900">
                    {prec.toFixed(1)}%
                  </td>
                );
              })}
              <td className="p-2 border-l border-slate-200 text-slate-400">
                —
              </td>
              <td className="p-2 border-l border-slate-200 font-bold text-brand-700">
                Acc: {totalClips > 0 ? ((labels.reduce((sum, l) => sum + (matrix[l] ? matrix[l][l] || 0 : 0), 0) / totalClips) * 100).toFixed(1) : 0}%
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}
