import os
import json

project_root = r"E:\SpeechPrj"
dev_json = os.path.join(project_root, "evaluation_results", "experimental", "dev_eval_results_30clips_refined.json")

with open(dev_json, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total dev clips: {len(data['predictions'])}")
for idx, p in enumerate(data['predictions']):
    words = p.get('words_list', [])
    text = p.get('pred_transcript', '')
    dur = p.get('duration_sec', 0.0)
    print(f"[{idx+1:02d}] {p['filename']} ({p['dataset']}) | Dur: {dur:.2f}s | Words: {len(words)} | GT_St: {p['gt_has_stutter']} | GT_Fl: {p['gt_has_filler']}")
    print(f"     Transcript: '{text}'")
    st_events = p.get('details', {}).get('stutter', [])
    print(f"     Stutter events: {len(st_events)} -> {[e['type'] for e in st_events]}")
    for e in st_events:
        if e['type'] == 'acoustic_prolongation':
            print(f"       -> Prolongation: dur={e['duration']}s, pitch_std={e.get('pitch_std_st')}, flux={e.get('mean_flux')}, start={e.get('start')}, end={e.get('end')}")
