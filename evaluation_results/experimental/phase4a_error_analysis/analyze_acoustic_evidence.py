import sys
import os
import json
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.experimental.config import AcousticConfig, StutterDetectorConfig
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.stutter_detector import HybridStutterDetector

inventory_path = Path('evaluation_results/experimental/phase4a_error_analysis/fn_detailed_inventory.json')
with open(inventory_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

extractor = AcousticFeatureExtractor(AcousticConfig())

analysis = []

for r in records:
    fn = r['filename']
    audio_path = r['audio_path']
    if not audio_path or not os.path.exists(audio_path):
        continue
        
    y, sr = extractor.load_audio(audio_path)
    audio_dur = len(y) / sr
    
    # Energy and voicing
    rms, noise_floor, vocal_thresh = extractor.compute_energy_and_noise_floor(y)
    frame_sec = extractor.config.hop_length_ms / 1000.0
    
    # Unvoiced / silent frames
    is_silent = rms < vocal_thresh
    
    # Detect continuous silence segments (pauses/blocks)
    silence_segments = []
    in_silence = False
    s_start = 0.0
    for i, s in enumerate(is_silent):
        t = i * frame_sec
        if s and not in_silence:
            in_silence = True
            s_start = t
        elif not s and in_silence:
            in_silence = False
            dur = t - s_start
            if dur >= 0.25: # >= 250ms
                silence_segments.append({'start': round(s_start, 2), 'end': round(t, 2), 'duration': round(dur, 2)})
    if in_silence:
        dur = (len(rms) * frame_sec) - s_start
        if dur >= 0.25:
            silence_segments.append({'start': round(s_start, 2), 'end': round(len(rms)*frame_sec, 2), 'duration': round(dur, 2)})

    # Initial silence before first voiced frame
    voiced_indices = np.where(rms >= vocal_thresh)[0]
    initial_silence = round(voiced_indices[0] * frame_sec, 2) if len(voiced_indices) > 0 else audio_dur
    
    detector = HybridStutterDetector(config=StutterDetectorConfig(), acoustic_extractor=extractor)
    n_ev, cands = detector.detect_acoustic_prolongations(y, sr)
    cand_summary = [{'start': c['start'], 'end': c['end'], 'duration': c['duration'], 'conf': c['confidence']} for c in cands]

    analysis.append({
        'filename': fn,
        'dataset': r['dataset'],
        'duration_sec': round(audio_dur, 2),
        'initial_silence_sec': initial_silence,
        'silence_segments_count': len(silence_segments),
        'silence_segments': silence_segments,
        'acoustic_prolongation_candidates': cand_summary,
        'raw_transcript': r['raw_transcript'],
        'annotations': r['annotation_details']
    })

out_json = Path('evaluation_results/experimental/phase4a_error_analysis/acoustic_evidence.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2)

print(f"Computed acoustic evidence for {len(analysis)} clips -> {out_json}")
for a in analysis:
    print(f"=== {a['filename']} ({a['dataset']}, {a['duration_sec']}s) ===")
    print(f"  Initial silence: {a['initial_silence_sec']}s")
    print(f"  Pauses >= 0.25s: {a['silence_segments']}")
    print(f"  Prolongation Cands: {a['acoustic_prolongation_candidates']}")
    print(f"  Annotations: {a['annotations']}")
    print(f"  Transcript: '{a['raw_transcript']}'")
    print()
