import React from 'react';
import { HelpCircle, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import StatusBadge from './StatusBadge';

export default function KpiCard({
  title,
  value,
  subtitle,
  delta,
  deltaType = 'positive', // 'positive' | 'negative' | 'neutral'
  badgeType,
  badgeText,
  tooltip,
  icon: Icon
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden group">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">
            {title}
            {tooltip && (
              <div className="relative group/tooltip inline-block cursor-help">
                <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-slate-600" />
                <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1.5 hidden group-hover/tooltip:block z-50 w-60 p-2 bg-slate-900 text-white text-xs rounded shadow-lg pointer-events-none">
                  {tooltip}
                </div>
              </div>
            )}
          </div>
          <div className="text-3xl font-bold tracking-tight text-slate-900 mt-1">
            {value}
          </div>
        </div>
        {Icon && (
          <div className="p-2.5 rounded-lg bg-slate-50 text-slate-600 border border-slate-100 group-hover:bg-brand-50 group-hover:text-brand-600 transition-colors">
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
        <div className="text-slate-500 truncate max-w-[180px]">
          {subtitle}
        </div>
        
        {delta && (
          <div className={`flex items-center font-semibold ${
            deltaType === 'positive' ? 'text-emerald-600' :
            deltaType === 'negative' ? 'text-rose-600' : 'text-slate-600'
          }`}>
            {deltaType === 'positive' ? <ArrowUpRight className="w-3.5 h-3.5 mr-0.5" /> :
             deltaType === 'negative' ? <ArrowDownRight className="w-3.5 h-3.5 mr-0.5" /> : null}
            {delta}
          </div>
        )}
      </div>

      {(badgeType || badgeText) && (
        <div className="mt-2.5">
          <StatusBadge type={badgeType} text={badgeText} />
        </div>
      )}
    </div>
  );
}
