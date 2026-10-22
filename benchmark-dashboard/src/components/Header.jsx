import React from 'react';
import { Filter } from 'lucide-react';

export default function Header({
  currentPageTitle,
  datasetFilter,
  setDatasetFilter,
  totalClipsCount,
  allClips = [],
}) {
  const totalCount = allClips.length || totalClipsCount || 0;
  const uclassCount = allClips.filter(c => c.dataset === 'UCLASS').length;
  const fbCount = allClips.filter(c => c.dataset === 'FluencyBank').length;
  const libriCount = allClips.filter(c => c.dataset === 'Libri_Stutter').length;

  const datasets = [
    { id: 'all', label: `Combined (${totalCount} clips)`, count: totalCount },
    { id: 'UCLASS', label: `UCLASS (${uclassCount || Math.round(totalCount / 3)})`, count: uclassCount },
    { id: 'FluencyBank', label: `FluencyBank (${fbCount || Math.round(totalCount / 3)})`, count: fbCount },
    { id: 'Libri_Stutter', label: `Libri_Stutter (${libriCount || Math.round(totalCount / 3)})`, count: libriCount },
  ];

  return (
    <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200 px-6 py-3.5 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            {currentPageTitle}
          </h1>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-brand-50 text-brand-700 border border-brand-200">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-600 animate-pulse"></span>
            N = {totalClipsCount} Clips
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-0.5">
          SpeechAI Multimodal Architecture Verification • {totalCount}-Clip Benchmark Evaluation
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {/* Global Dataset Filter */}
        <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs font-medium">
          <span className="px-2 text-slate-400 flex items-center gap-1">
            <Filter className="w-3.5 h-3.5" />
            Dataset:
          </span>
          {datasets.map(d => (
            <button
              key={d.id}
              onClick={() => setDatasetFilter(d.id)}
              className={`px-2.5 py-1 rounded-md transition-all ${
                datasetFilter === d.id
                  ? 'bg-white text-brand-700 font-semibold shadow-xs border border-slate-200/60'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'
              }`}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
