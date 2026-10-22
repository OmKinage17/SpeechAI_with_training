import React from 'react';

export default function StatusBadge({ type, text, className = '' }) {
  const badgeStyles = {
    groundTruth: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    goldRef: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    proxy: 'bg-amber-50 text-amber-700 border-amber-200',
    audit: 'bg-purple-50 text-purple-700 border-purple-200',
    omitted: 'bg-slate-100 text-slate-600 border-slate-300',
    warning: 'bg-rose-50 text-rose-700 border-rose-200',
  };

  const labels = {
    groundTruth: 'Supervised Ground Truth',
    goldRef: 'Gold Verbatim Reference',
    proxy: 'Signal Processing Proxy',
    audit: 'Deterministic Rule Audit',
    omitted: 'Unannotated / Schema-Omitted',
    warning: 'Caution / Limitation',
  };

  const style = badgeStyles[type] || badgeStyles.omitted;
  const labelText = text || labels[type] || type;

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${style} ${className}`}>
      <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-current opacity-70"></span>
      {labelText}
    </span>
  );
}
