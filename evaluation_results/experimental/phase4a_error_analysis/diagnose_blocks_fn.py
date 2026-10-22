import sys
import os
import json
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.experimental.config import StutterDetectorConfig, AcousticConfig
from backend.experimental.acoustic_features import AcousticFeatureExtractor
from backend.experimental.stutter_detector import HybridStutterDetector

inventory_path = Path('evaluation_results/experimental/phase4a_error_analysis/fn_detailed_inventory.json')
with open(inventory_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

acoustic = AcousticFeatureExtractor(AcousticConfig())
detector = HybridStutterDetector(config=StutterDetectorConfig(), acoustic_extractor=acoustic)

print(f"{'Filename':<35} | {'Dur':<5} | {'1stWordStart':<12} | {'InterGaps':<20} | {'Why Block Missed'}")
print("-" * 105)

for r in records:
    fn = r['filename']
    audio_path = r['audio_path']
    words_list = r['words_list']
    dur = f"{r['duration']:.2f}s" if r['duration'] else "N/A"
    
    if not words_list:
        print(f"{fn:<35} | {dur:<5} | {'N/A (Empty)':<12} | {'N/A':<20} | Empty words_list (Whisper failed/empty)")
        continue
        
    first_start = float(words_list[0].get('start', 0.0))
    
    # Calculate inter-word gaps
    gaps = []
    for i in range(len(words_list) - 1):
        g = float(words_list[i+1].get('start', 0.0)) - float(words_list[i].get('end', 0.0))
        if g >= 0.20:
            gaps.append(round(g, 2))
            
    gaps_str = str(gaps) if gaps else "None >= 0.20s"
    
    # Let's inspect block detection details
    reason = []
    if audio_path and os.path.exists(audio_path):
        y, sr = acoustic.load_audio(audio_path)
        rms, noise_floor, vocal_thresh = acoustic.compute_energy_and_noise_floor(y)
        frame_sec = acoustic.config.hop_length_ms / 1000.0
        
        # Check initial block
        if first_start >= detector.config.min_block_silence_sec: # 0.40s
            first_frame = int(first_start / frame_sec)
            if first_frame < len(rms):
                burst_energy = float(np.max(rms[first_frame:min(len(rms), first_frame + 4)]))
                pre_energy = float(np.mean(rms[:first_frame])) if first_frame > 0 else 0.0
                ratio = burst_energy / (pre_energy + 1e-4)
                if burst_energy < vocal_thresh:
                    reason.append(f"Initial gap {first_start:.2f}s: burst_energy ({burst_energy:.4f}) < vocal_thresh ({vocal_thresh:.4f})")
                elif ratio < detector.config.plosive_burst_energy_ratio:
                    reason.append(f"Initial gap {first_start:.2f}s: burst ratio {ratio:.2f} < 3.0 (pre_energy={pre_energy:.4f})")
                else:
                    reason.append(f"Initial gap {first_start:.2f}s: Passed block criteria!")
        else:
            reason.append(f"1stWordStart={first_start:.2f}s < min_block_silence (0.40s)")
            
        # Check inter gaps
        for i in range(len(words_list) - 1):
            curr_end = float(words_list[i].get("end", 0.0))
            next_start = float(words_list[i+1].get("start", 0.0))
            gap = next_start - curr_end
            if gap >= detector.config.min_block_silence_sec:
                margin_sec = 0.03
                s_frame = int((curr_end + margin_sec) / frame_sec)
                e_frame = int((next_start - margin_sec) / frame_sec)
                burst_start_frame = int(next_start / frame_sec)
                if e_frame > s_frame and burst_start_frame < len(rms):
                    gap_rms = float(np.mean(rms[s_frame:e_frame]))
                    burst_rms = float(np.max(rms[burst_start_frame:min(len(rms), burst_start_frame + 4)]))
                    if gap_rms > max(0.03, vocal_thresh * 1.5):
                        reason.append(f"Gap {gap:.2f}s: gap_rms ({gap_rms:.4f}) > vocal_thresh*1.5 ({max(0.03, vocal_thresh * 1.5):.4f}) [noisy silence]")
                    elif burst_rms < 2.5 * (gap_rms + 1e-4):
                        reason.append(f"Gap {gap:.2f}s: burst/gap ratio ({burst_rms/(gap_rms+1e-4):.2f}) < 2.5")
                    elif burst_rms < vocal_thresh:
                        reason.append(f"Gap {gap:.2f}s: burst_rms < vocal_thresh")
                    else:
                        reason.append(f"Gap {gap:.2f}s: Passed inter block criteria!")
    else:
        reason.append("Audio file not found")
        
    print(f"{fn:<35} | {dur:<5} | {first_start:<12.2f} | {gaps_str:<20} | {'; '.join(reason)}")
