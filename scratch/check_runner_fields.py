import os, re

runner = r'E:\SpeechPrj\backend\experimental\run_phase5_paired_benchmark.py'
if not os.path.exists(runner):
    print("Runner not found:", runner)
else:
    with open(runner, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find all gt_ field accesses
    gt_reads = sorted(set(re.findall(r"'(gt_[a-z_]+)'", content)))
    print("gt_ fields referenced in runner:")
    for g in gt_reads:
        print(" ", g)
    print()
    print("References gt_phrase_rep:", ('gt_phrase_rep' in content))
    print("References gt_has_stutter:", ('gt_has_stutter' in content))
    print("References gt_dominant:", ('gt_dominant' in content))
    # Check what the evaluator expects
    print()
    print("Total lines:", content.count('\n'))
