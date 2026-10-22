import React, { useState, useMemo } from 'react';
import {
  Search, Filter, Download, CheckCircle, XCircle, ChevronRight,
  Eye, X, MessageSquare, Gauge, PauseCircle, Activity
} from 'lucide-react';

export default function ClipExplorer({ clips, onExportCsv }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [datasetFilter, setDatasetFilter] = useState('all');
  const [gtClassFilter, setGtClassFilter] = useState('all');
  const [predClassFilter, setPredClassFilter] = useState('all');
  const [correctFilter, setCorrectFilter] = useState('all'); // 'all' | 'correct' | 'incorrect'
  const [errorCategoryFilter, setErrorCategoryFilter] = useState('all');
  const [selectedClip, setSelectedClip] = useState(null);

  // Filtered clips
  const filteredClips = useMemo(() => {
    return clips.filter(c => {
      // Search
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        const matchesName = c.filename.toLowerCase().includes(term);
        const matchesTranscript = (c.transcript || '').toLowerCase().includes(term);
        const matchesSpeaker = (c.speaker_id || '').toLowerCase().includes(term);
        if (!matchesName && !matchesTranscript && !matchesSpeaker) return false;
      }
      // Dataset
      if (datasetFilter !== 'all' && c.dataset !== datasetFilter) return false;
      // GT Class
      if (gtClassFilter !== 'all' && c.gt_dominant !== gtClassFilter) return false;
      // Pred Class
      if (predClassFilter !== 'all' && c.pred_dominant !== predClassFilter) return false;
      // Correctness
      if (correctFilter === 'correct' && !c.is_correct_multiclass) return false;
      if (correctFilter === 'incorrect' && c.is_correct_multiclass) return false;
      // Error Category
      if (errorCategoryFilter !== 'all' && c.error_type !== errorCategoryFilter) return false;

      return true;
    });
  }, [clips, searchTerm, datasetFilter, gtClassFilter, predClassFilter, correctFilter, errorCategoryFilter]);

  // Export filtered CSV handler
  const handleExportFiltered = () => {
    const headers = [
      'Dataset', 'Filename', 'SpeakerID', 'DurationSec',
      'GT_Dominant', 'GT_Stutter', 'GT_Filler', 'GT_Block',
      'Pred_Dominant', 'Pred_Stutter', 'Pred_Filler', 'Pred_Block',
      'Correct_Multiclass', 'Correct_BinaryStutter', 'ErrorType',
      'StutterEvents', 'LexicalReps', 'Prolongations', 'Blocks',
      'FillerCount', 'WPM', 'RefWPM', 'WER', 'LatencySec'
    ];

    const rows = filteredClips.map(c => [
      c.dataset, c.filename, c.speaker_id, c.clip_duration_sec,
      c.gt_dominant, c.gt_has_stutter ? 1 : 0, c.gt_has_filler ? 1 : 0, c.gt_block,
      c.pred_dominant, c.pred_has_stutter ? 1 : 0, c.pred_has_filler ? 1 : 0, c.pred_block,
      c.is_correct_multiclass ? 1 : 0, c.is_correct_binary_stutter ? 1 : 0, c.error_type,
      c.stutter_events, c.lex_reps, c.prolongations, c.blocks,
      c.filler_count, c.wpm, c.ref_wpm || '', c.wer !== null ? c.wer : '', c.latency_sec
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.map(v => `"${v}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `speechai_filtered_clips_${filteredClips.length}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            Per-Clip Explorer & Error Diagnosis Grid
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Searchable, filterable inspection of all {clips.length} validation clips with multimodal outputs and error classifications.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 font-medium">
            Showing <strong>{filteredClips.length}</strong> of {clips.length} clips
          </span>
          <button
            onClick={handleExportFiltered}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-brand-600 hover:bg-brand-700 text-white rounded-lg text-xs font-semibold shadow-xs transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Export Filtered CSV
          </button>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          {/* Search Input */}
          <div className="lg:col-span-2 relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5 pointer-events-none" />
            <input
              type="text"
              placeholder="Search filename, speaker, transcript..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 border border-slate-200 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
            />
          </div>

          {/* Dataset Filter */}
          <div>
            <select
              value={datasetFilter}
              onChange={e => setDatasetFilter(e.target.value)}
              className="w-full py-1.5 px-2.5 border border-slate-200 rounded-lg text-xs bg-slate-50/50 text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <option value="all">All Datasets ({clips.length})</option>
              <option value="UCLASS">UCLASS ({clips.filter(c => c.dataset === 'UCLASS').length})</option>
              <option value="FluencyBank">FluencyBank ({clips.filter(c => c.dataset === 'FluencyBank').length})</option>
              <option value="Libri_Stutter">Libri_Stutter ({clips.filter(c => c.dataset === 'Libri_Stutter').length})</option>
            </select>
          </div>

          {/* GT Class Filter */}
          <div>
            <select
              value={gtClassFilter}
              onChange={e => setGtClassFilter(e.target.value)}
              className="w-full py-1.5 px-2.5 border border-slate-200 rounded-lg text-xs bg-slate-50/50 text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <option value="all">All GT Classes</option>
              <option value="Clean">GT Clean ({clips.filter(c => c.gt_dominant === 'Clean').length})</option>
              <option value="Stutter">GT Stutter ({clips.filter(c => c.gt_dominant === 'Stutter').length})</option>
              <option value="Filler">GT Filler ({clips.filter(c => c.gt_dominant === 'Filler').length})</option>
            </select>
          </div>

          {/* Correctness Filter */}
          <div>
            <select
              value={correctFilter}
              onChange={e => setCorrectFilter(e.target.value)}
              className="w-full py-1.5 px-2.5 border border-slate-200 rounded-lg text-xs bg-slate-50/50 text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <option value="all">All Results ({clips.length})</option>
              <option value="correct">Correct Only ({clips.filter(c => c.correct).length})</option>
              <option value="incorrect">Errors Only ({clips.filter(c => !c.correct).length})</option>
            </select>
          </div>

          {/* Error Category Filter */}
          <div>
            <select
              value={errorCategoryFilter}
              onChange={e => setErrorCategoryFilter(e.target.value)}
              className="w-full py-1.5 px-2.5 border border-slate-200 rounded-lg text-xs bg-slate-50/50 text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <option value="all">All Error Types</option>
              <option value="Correct">Correct Predictions</option>
              <option value="False Alarm (Stutter)">False Alarm (Stutter)</option>
              <option value="False Alarm (Filler)">False Alarm (Filler)</option>
              <option value="Missed Stutter">Missed Stutter</option>
              <option value="Missed Filler">Missed Filler</option>
              <option value="Stutter Confused as Filler">Stutter Confused as Filler</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-sm overflow-hidden">
        <div className="overflow-x-auto max-h-[600px]">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="sticky top-0 bg-slate-100/90 backdrop-blur text-slate-700 font-semibold border-b border-slate-200 z-10">
              <tr>
                <th className="p-3">Corpus</th>
                <th className="p-3">Filename / Clip ID</th>
                <th className="p-3">GT Dominant</th>
                <th className="p-3">Predicted</th>
                <th className="p-3 text-center">Correct?</th>
                <th className="p-3">Events Breakdown</th>
                <th className="p-3">WPM</th>
                <th className="p-3">Pauses</th>
                <th className="p-3">Confidence</th>
                <th className="p-3 text-center">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredClips.map((c, i) => {
                const isCorrect = c.is_correct_multiclass;
                return (
                  <tr key={i} className="hover:bg-slate-50/80 transition-colors">
                    <td className="p-3 font-medium text-slate-800">{c.dataset}</td>
                    <td className="p-3 font-mono text-[11px] text-slate-600 max-w-[170px] truncate" title={c.filename}>
                      {c.filename}
                    </td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        c.gt_dominant === 'Stutter' ? 'bg-rose-100 text-rose-800' :
                        c.gt_dominant === 'Filler' ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {c.gt_dominant}
                      </span>
                    </td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        c.pred_dominant === 'Stutter' ? 'bg-rose-100 text-rose-800' :
                        c.pred_dominant === 'Filler' ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {c.pred_dominant}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      {isCorrect ? (
                        <span className="inline-flex items-center gap-1 text-emerald-600 font-semibold text-[11px]">
                          <CheckCircle className="w-3.5 h-3.5" /> OK
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-rose-600 font-semibold text-[11px]">
                          <XCircle className="w-3.5 h-3.5" /> {c.error_type}
                        </span>
                      )}
                    </td>
                    <td className="p-3 font-mono text-slate-600">
                      R:{c.lex_reps} | P:{c.prolongations} | B:{c.blocks} | F:{c.filler_count}
                    </td>
                    <td className="p-3 font-semibold text-slate-800">
                      {c.wpm}
                    </td>
                    <td className="p-3 text-slate-600">
                      {c.pause_count} ({c.total_pause_time_sec}s)
                    </td>
                    <td className="p-3 font-mono text-slate-600">
                      {(c.confidence * 100).toFixed(0)}%
                    </td>
                    <td className="p-3 text-center">
                      <button
                        onClick={() => setSelectedClip(c)}
                        className="p-1.5 rounded hover:bg-brand-50 text-slate-500 hover:text-brand-600 transition-colors"
                        title="View Full Clip Details"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal Inspector Drawer for Single Clip */}
      {selectedClip && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-slate-200 p-6 space-y-5 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-start justify-between border-b border-slate-100 pb-3">
              <div>
                <span className="text-[10px] uppercase font-semibold text-brand-600 tracking-wider">
                  Clip Detail Inspector • {selectedClip.dataset}
                </span>
                <h3 className="text-lg font-bold text-slate-900 font-mono mt-0.5">
                  {selectedClip.filename}
                </h3>
              </div>
              <button
                onClick={() => setSelectedClip(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Classification Summary */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Ground Truth</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{selectedClip.gt_dominant}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Predicted</span>
                <span className="text-sm font-bold text-brand-700 block mt-0.5">{selectedClip.pred_dominant}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Confidence</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{(selectedClip.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Pacing</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{selectedClip.wpm} WPM</span>
              </div>
            </div>

            {/* Transcripts */}
            <div className="space-y-3">
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block">
                  ASR Recognized Hypothesis Transcript:
                </span>
                <p className="text-xs text-slate-800 italic">
                  "{selectedClip.transcript || 'No speech recognized / silent window'}"
                </p>
              </div>

              {selectedClip.reference && (
                <div className="p-3 rounded-lg bg-indigo-50/50 border border-indigo-200 space-y-1">
                  <span className="text-[10px] font-semibold text-indigo-700 uppercase tracking-wider block">
                    Published Gold Reference Book Transcript:
                  </span>
                  <p className="text-xs text-indigo-950 font-serif">
                    "{selectedClip.reference}"
                  </p>
                </div>
              )}
            </div>

            {/* Multimodal Event Counts */}
            <div className="border border-slate-200 rounded-lg p-3 text-xs space-y-2">
              <span className="font-semibold text-slate-800 block">Acoustic & Lexical Dysfluency Subtypes:</span>
              <div className="grid grid-cols-4 gap-2 text-center">
                <div className="p-2 rounded bg-slate-50">Lexical Reps: <strong>{selectedClip.lex_reps}</strong></div>
                <div className="p-2 rounded bg-slate-50">Prolongations: <strong>{selectedClip.prolongations}</strong></div>
                <div className="p-2 rounded bg-slate-50">Acoustic Blocks: <strong>{selectedClip.blocks}</strong></div>
                <div className="p-2 rounded bg-slate-50">Filler Words: <strong>{selectedClip.filler_count}</strong></div>
              </div>
            </div>

            {/* Coaching Feedback Items */}
            {selectedClip.coaching_items && selectedClip.coaching_items.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-800 block">Generated Coaching Feedback Advice:</span>
                <div className="space-y-1.5 text-xs">
                  {selectedClip.coaching_items.map((fb, idx) => (
                    <div key={idx} className="p-2 rounded-lg bg-slate-50 border border-slate-200 flex items-start gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold shrink-0 ${
                        fb.type === 'Warning' ? 'bg-rose-100 text-rose-800' :
                        fb.type === 'Caution' ? 'bg-amber-100 text-amber-800' :
                        fb.type === 'Info' ? 'bg-blue-100 text-blue-800' : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {fb.category}
                      </span>
                      <span className="text-slate-700">{fb.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
