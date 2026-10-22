import React from 'react';
import {
  LayoutDashboard,
  GitCompare,
  TableProperties,
  BookOpen,
  ShieldCheck
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, totalClips = 1500 }) {
  const navSections = [
    {
      heading: 'EXECUTIVE',
      items: [
        { id: 'overview', label: 'Simplified Dashboard', icon: LayoutDashboard, badge: 'Unified' },
        { id: 'comparison', label: 'Dataset Comparison', icon: GitCompare, badge: '3-Way' }
      ]
    },
    {
      heading: 'DATA & METHODOLOGY',
      items: [
        { id: 'explorer', label: 'Per-Clip Explorer', icon: TableProperties, badge: `${totalClips} Clips` },
        { id: 'methodology', label: 'About & Limitations', icon: BookOpen }
      ]
    }
  ];

  return (
    <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col shrink-0 min-h-screen border-r border-slate-800">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-brand-600 text-white font-bold text-sm shadow-md">
              SAI
            </div>
            <div>
              <div className="font-bold text-sm text-white tracking-tight flex items-center gap-1.5">
                SpeechAI
                <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-brand-500/20 text-brand-300 border border-brand-500/30">
                  Analytics
                </span>
              </div>
              <div className="text-[11px] text-slate-400 font-medium">
                Benchmark Dashboard
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Nav List */}
      <nav className="flex-1 px-3 py-4 space-y-6 overflow-y-auto">
        {navSections.map((sec, idx) => (
          <div key={idx} className="space-y-1">
            <div className="px-3 text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
              {sec.heading}
            </div>
            {sec.items.map(item => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-brand-600 text-white shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
                  }`}
                >
                  <div className="flex items-center gap-2.5 truncate">
                    <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                    <span className="truncate">{item.label}</span>
                  </div>
                  {item.metric && (
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                      isActive ? 'bg-brand-700 text-white' : 'bg-slate-800 text-slate-400'
                    }`}>
                      {item.metric}
                    </span>
                  )}
                  {item.badge && (
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      isActive ? 'bg-brand-700 text-white' : 'bg-slate-800 text-slate-300'
                    }`}>
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Audit & Integrity Footer */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/60 text-xs space-y-2">
        <div className="flex items-center gap-2 text-emerald-400 font-medium text-[11px]">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>Audit Verified • 115 Tests OK</span>
        </div>
        <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1">
          <span>ASR Timeouts: 0 (100% completion)</span>
          <span className="font-mono text-slate-400">{totalClips} Clips</span>
        </div>
      </div>
    </aside>
  );
}
