import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

export default function TooltipHelp({ term, text }) {
  const [show, setShow] = useState(false);

  const termDefinitions = {
    'F1 Score': 'Harmonic mean of precision and recall. Balances false alarms and missed detections (1.0 is perfect).',
    'Precision': 'Of all positive detections made by the model, the percentage that were truly positive.',
    'Recall': 'Of all actual positive events, the percentage correctly identified by the model (Sensitivity).',
    'Specificity': 'True negative rate: ability to correctly reject fluent or control audio without false alarms.',
    'MAE': 'Mean Absolute Error: average magnitude of error in Words Per Minute between prediction and reference.',
    'RMSE': 'Root Mean Squared Error: standard deviation of residuals; penalizes large prediction errors.',
    'Pearson r': 'Linear correlation between predicted and reference values (1.0 = perfect positive association).',
    'WER': 'Word Error Rate: percentage of words altered via substitutions, deletions, or insertions in ASR.',
    'CER': 'Character Error Rate: percentage of character-level edits, reflecting phonetic transcription fidelity.',
    'Compliance': 'Software consistency audit verifying that 100% of rule triggers executed deterministically without exceptions.',
    'Realistic WPM': 'Human conversational speech rates range between 100-220 WPM. Values >250 WPM are typically 3s extrapolation or ASR loop artifacts.',
  };

  const explanation = text || termDefinitions[term] || term;

  return (
    <span 
      className="relative inline-flex items-center ml-1 text-slate-400 hover:text-slate-700 cursor-pointer select-none"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onClick={() => setShow(!show)}
      title={explanation}
    >
      <HelpCircle className="w-3.5 h-3.5" />
      {show && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-56 p-2 bg-slate-900 text-white text-[11px] font-normal rounded-lg shadow-xl z-50 pointer-events-none leading-relaxed text-center">
          <strong className="block font-semibold text-slate-200 mb-0.5">{term}</strong>
          {explanation}
        </span>
      )}
    </span>
  );
}
