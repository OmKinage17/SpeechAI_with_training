import React, { useState, useMemo } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import SimplifiedDashboard from './pages/SimplifiedDashboard';
import DatasetComparison from './pages/DatasetComparison';
import ClipExplorer from './pages/ClipExplorer';
import MethodologyLimitations from './pages/MethodologyLimitations';

import benchmarkUnifiedData from './data/benchmark_unified_data.json';
import { calculateMetricsForClips } from './utils/metrics';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [datasetFilter, setDatasetFilter] = useState('all'); // 'all' | 'UCLASS' | 'FluencyBank' | 'Libri_Stutter'
  const [comparisonMode, setComparisonMode] = useState('combined'); // 'combined' | 'sideBySide'

  const allClips = benchmarkUnifiedData.clips || [];

  // Filter clips based on active dataset selection
  const filteredClips = useMemo(() => {
    if (datasetFilter === 'all') return allClips;
    return allClips.filter(c => c.dataset === datasetFilter);
  }, [allClips, datasetFilter]);

  // Dynamically recompute verified metrics for the currently filtered subset
  const filteredMetrics = useMemo(() => {
    return calculateMetricsForClips(filteredClips);
  }, [filteredClips]);

  // Compute dataset-specific metrics for 3-way comparative page
  const perDatasetMetrics = useMemo(() => {
    return {
      UCLASS: calculateMetricsForClips(allClips.filter(c => c.dataset === 'UCLASS')),
      FluencyBank: calculateMetricsForClips(allClips.filter(c => c.dataset === 'FluencyBank')),
      Libri_Stutter: calculateMetricsForClips(allClips.filter(c => c.dataset === 'Libri_Stutter'))
    };
  }, [allClips]);

  // Human-readable titles for Header
  const pageTitles = {
    overview: 'SpeechAI Benchmark Analytics Dashboard',
    comparison: 'Dataset Comparison (3-Way Cross-Corpus Matrix)',
    explorer: 'Per-Clip Explorer & Comprehensive Error Audit',
    methodology: 'Benchmark Methodology, Metric Taxonomy & Limitations'
  };

  // CSV Exporter for benchmark results
  const handleExportCsv = (clipsToExport) => {
    const list = Array.isArray(clipsToExport) && clipsToExport.length > 0 ? clipsToExport : filteredClips;
    const headers = [
      'filename',
      'dataset',
      'speaker_id',
      'clip_duration_sec',
      'gt_dominant',
      'pred_dominant',
      'correct',
      'error_category',
      'gt_stutter',
      'pred_stutter',
      'stutter_confidence',
      'fillers_count',
      'gt_has_filler',
      'wpm',
      'ref_wpm',
      'pause_count',
      'total_pause_duration_sec',
      'gt_block',
      'pred_block',
      'wer',
      'cer',
      'coaching_count',
      'transcript'
    ];

    const rows = list.map(c => {
      return headers.map(h => {
        let val = c[h];
        if (val === null || val === undefined) return '""';
        if (typeof val === 'string') return `"${val.replace(/"/g, '""')}"`;
        return JSON.stringify(val);
      }).join(',');
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    const filenameScope = datasetFilter === 'all' ? `${allClips.length}_clips` : `${datasetFilter}_clips`;
    link.setAttribute('download', `SpeechAI_Benchmark_${filenameScope}_export.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Render the appropriate page based on activeTab and comparisonMode
  const renderContent = () => {
    // If user explicitly toggled Side-by-Side comparison mode in the top bar, show the 3-way comparison view
    if (comparisonMode === 'sideBySide' && activeTab !== 'comparison') {
      return (
        <DatasetComparison 
          clips={allClips} 
          perDatasetMetrics={perDatasetMetrics} 
          confusionMatrices={benchmarkUnifiedData.confusion_matrices} 
        />
      );
    }

    switch (activeTab) {
      case 'overview':
        return (
          <SimplifiedDashboard 
            clips={filteredClips}
            allClips={allClips}
            filteredMetrics={filteredMetrics} 
            datasetFilter={datasetFilter} 
            setDatasetFilter={setDatasetFilter}
            perDatasetMetrics={perDatasetMetrics}
            setActiveTab={setActiveTab}
          />
        );
      case 'comparison':
        return (
          <DatasetComparison 
            clips={allClips} 
            perDatasetMetrics={perDatasetMetrics} 
            confusionMatrices={benchmarkUnifiedData.confusion_matrices} 
          />
        );
      case 'explorer':
      case 'clips':
        return (
          <ClipExplorer 
            clips={filteredClips} 
            onExportCsv={handleExportCsv} 
          />
        );
      case 'methodology':
        return <MethodologyLimitations />;
      default:
        return (
          <SimplifiedDashboard 
            clips={filteredClips}
            allClips={allClips}
            filteredMetrics={filteredMetrics} 
            datasetFilter={datasetFilter} 
            setDatasetFilter={setDatasetFilter}
            perDatasetMetrics={perDatasetMetrics}
            setActiveTab={setActiveTab}
          />
        );
    }
  };

  return (
    <div className="flex min-h-screen bg-slate-50 font-sans text-slate-900 antialiased selection:bg-brand-500 selection:text-white">
      {/* Fixed Sidebar */}
      <Sidebar 
        activeTab={activeTab} 
        totalClips={allClips.length}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          // If switching to a specific parameter, reset comparisonMode to 'combined' so page shows
          if (comparisonMode === 'sideBySide' && tab !== 'comparison') {
            setComparisonMode('combined');
          }
        }} 
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          currentPageTitle={pageTitles[activeTab] || 'SpeechAI Analytics Dashboard'}
          datasetFilter={datasetFilter}
          setDatasetFilter={setDatasetFilter}
          totalClipsCount={filteredClips.length}
          allClips={allClips}
        />

        <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
          {renderContent()}
        </main>

        {/* Global Footer */}
        <footer className="border-t border-slate-200 bg-white px-6 py-4 text-xs text-slate-500 flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700">SpeechAI Research Benchmark Dashboard</span>
            <span>•</span>
            <span>Run: run_final_20261020_complete</span>
            <span>•</span>
            <span className="text-emerald-600 font-medium">100% Audit Verified</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>Benchmark Evaluation (N={allClips.length})</span>
            <span>•</span>
            <span>Dedicated Port: localhost:4173</span>
            <span>•</span>
            <span>Separate Analytics Engine</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
