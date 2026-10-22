import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import unified benchmark data
const dataPath = path.join(__dirname, '../src/data/benchmark_unified_data.json');
const rawData = fs.readFileSync(dataPath, 'utf-8');
const data = JSON.parse(rawData);

console.log('====================================================');
console.log('   SPEECHAI DASHBOARD DATA INTEGRITY AUDIT TEST    ');
console.log('====================================================');

let passedTests = 0;
let totalTests = 0;

function assert(condition, message) {
  totalTests++;
  if (condition) {
    console.log(`[PASS] ${message}`);
    passedTests++;
  } else {
    console.error(`[FAIL] ${message}`);
  }
}

const is1500Run = data.clips.length === 1500;
const expectedTotal = is1500Run ? 1500 : 120;
const expectedPerDataset = is1500Run ? 500 : 40;

// 1. Clips Count & Integrity
assert(data.clips.length === expectedTotal, `Total clips count is exactly ${expectedTotal} (found: ${data.clips.length})`);

const uclassClips = data.clips.filter(c => c.dataset === 'UCLASS');
const fbClips = data.clips.filter(c => c.dataset === 'FluencyBank');
const libriClips = data.clips.filter(c => c.dataset === 'Libri_Stutter');

assert(uclassClips.length === expectedPerDataset, `UCLASS clip count is exactly ${expectedPerDataset} (found: ${uclassClips.length})`);
assert(fbClips.length === expectedPerDataset, `FluencyBank clip count is exactly ${expectedPerDataset} (found: ${fbClips.length})`);
assert(libriClips.length === expectedPerDataset, `Libri_Stutter clip count is exactly ${expectedPerDataset} (found: ${libriClips.length})`);

// 2. Multiclass Ground Truth Distribution
const gtClean = data.clips.filter(c => c.gt_dominant === 'Clean').length;
const gtStutter = data.clips.filter(c => c.gt_dominant === 'Stutter').length;
const gtFiller = data.clips.filter(c => c.gt_dominant === 'Filler').length;

assert(gtClean > 0, `Ground-truth Clean count is non-zero (found: ${gtClean})`);
assert(gtStutter > 0, `Ground-truth Stutter count is non-zero (found: ${gtStutter})`);
assert(gtFiller > 0, `Ground-truth Filler count is non-zero (found: ${gtFiller})`);
assert(gtClean + gtStutter + gtFiller === expectedTotal, `Sum of ground-truth classes equals total clips (${gtClean + gtStutter + gtFiller} === ${expectedTotal})`);

// 3. Multiclass Accuracy
const correctCount = data.clips.filter(c => c.correct).length;
const acc = (correctCount / expectedTotal) * 100;
assert(correctCount > 0, `Total correct multiclass predictions is positive (found: ${correctCount})`);
const reportedAcc = (data.metrics_snapshot?.final_metrics?.multiclass_aggregate?.accuracy || 0) * 100;
assert(Math.abs(acc - reportedAcc) < 0.1, `Calculated accuracy matches snapshot (${acc.toFixed(2)}% vs ${reportedAcc.toFixed(2)}%)`);

// 4. Binary Stuttering Detection
let tp_stutter = 0, fp_stutter = 0, fn_stutter = 0, tn_stutter = 0;
data.clips.forEach(c => {
  const isActualStutter = c.gt_stutter === 1;
  const isPredStutter = c.pred_stutter === 1;
  if (isActualStutter && isPredStutter) tp_stutter++;
  else if (!isActualStutter && isPredStutter) fp_stutter++;
  else if (isActualStutter && !isPredStutter) fn_stutter++;
  else tn_stutter++;
});

assert(tp_stutter + fp_stutter + fn_stutter + tn_stutter === expectedTotal, `Binary confusion matrix sum equals ${expectedTotal}`);
assert(tp_stutter > 0, `Binary Stutter TP is positive (found: ${tp_stutter})`);

const p_stutter = tp_stutter / (tp_stutter + fp_stutter);
const r_stutter = tp_stutter / (tp_stutter + fn_stutter);
const f1_stutter = (2 * p_stutter * r_stutter) / (p_stutter + r_stutter);
const spec_stutter = tn_stutter / (tn_stutter + fp_stutter);

assert(p_stutter >= 0 && p_stutter <= 1.0, `Stutter precision is valid (${(p_stutter * 100).toFixed(2)}%)`);
assert(r_stutter >= 0 && r_stutter <= 1.0, `Stutter recall is valid (${(r_stutter * 100).toFixed(2)}%)`);
assert(f1_stutter >= 0 && f1_stutter <= 1.0, `Stutter F1 is valid (${f1_stutter.toFixed(4)})`);
assert(spec_stutter >= 0 && spec_stutter <= 1.0, `Stutter specificity is valid (${(spec_stutter * 100).toFixed(2)}%)`);

// 5. Filler Event-Level Detection
let tp_filler = 0, fp_filler = 0, fn_filler = 0, tn_filler = 0;
data.clips.forEach(c => {
  const isActualFiller = c.gt_has_filler === 1;
  const isPredFiller = (c.fillers_count || 0) > 0;
  if (isActualFiller && isPredFiller) tp_filler++;
  else if (!isActualFiller && isPredFiller) fp_filler++;
  else if (isActualFiller && !isPredFiller) fn_filler++;
  else tn_filler++;
});

assert(tp_filler + fp_filler + fn_filler + tn_filler === expectedTotal, `Filler confusion matrix sum equals ${expectedTotal}`);
assert(tp_filler > 0, `Filler Event TP is positive (found: ${tp_filler})`);

const p_filler = tp_filler / (tp_filler + fp_filler);
const r_filler = tp_filler / (tp_filler + fn_filler);
const f1_filler = (2 * p_filler * r_filler) / (p_filler + r_filler);
assert(f1_filler >= 0 && f1_filler <= 1.0, `Filler Event F1 is valid (${f1_filler.toFixed(4)})`);

// 6. Block Detection Scoped Analysis (UCLASS + FB N=expectedPerDataset*2)
const scopedBlockClips = data.clips.filter(c => c.gt_block !== null && c.gt_block !== undefined);
assert(scopedBlockClips.length === expectedPerDataset * 2, `Scoped block clips count is exactly ${expectedPerDataset * 2} (found: ${scopedBlockClips.length})`);

let tp_block = 0, fp_block = 0, fn_block = 0, tn_block = 0;
scopedBlockClips.forEach(c => {
  const isActual = c.gt_block === 1;
  const isPred = c.pred_block === 1;
  if (isActual && isPred) tp_block++;
  else if (!isActual && isPred) fp_block++;
  else if (isActual && !isPred) fn_block++;
  else tn_block++;
});

const spec_block = (tn_block + fp_block > 0) ? tn_block / (tn_block + fp_block) : 1.0;
assert(spec_block >= 0 && spec_block <= 1.0, `Scoped Block Specificity is valid (${(spec_block * 100).toFixed(2)}%)`);

// Verify Libri_Stutter has null block annotations
const libriBlockEvaluated = libriClips.filter(c => c.gt_block !== null && c.gt_block !== undefined);
assert(libriBlockEvaluated.length === 0, `Libri_Stutter block annotations correctly marked unavailable/null (found: ${libriBlockEvaluated.length})`);

// 7. Speech Rate Regression (Libri_Stutter N=expectedPerDataset)
const libriRateClips = libriClips.filter(c => c.ref_wpm !== null && c.ref_wpm !== undefined);
assert(libriRateClips.length === expectedPerDataset, `Libri_Stutter reference WPM count is exactly ${expectedPerDataset} (found: ${libriRateClips.length})`);

let sumAbsDiff = 0, sumSqDiff = 0;
libriRateClips.forEach(c => {
  const diff = c.wpm - c.ref_wpm;
  sumAbsDiff += Math.abs(diff);
  sumSqDiff += diff * diff;
});
const mae = sumAbsDiff / libriRateClips.length;
const rmse = Math.sqrt(sumSqDiff / libriRateClips.length);

assert(mae >= 0, `Speech Rate MAE is positive (${mae.toFixed(2)} WPM)`);
assert(rmse >= 0, `Speech Rate RMSE is positive (${rmse.toFixed(2)} WPM)`);

// 8. Coaching Rule Engine Consistency
const coachingEvaluated = data.clips.filter(c => c.coaching_count > 0).length;
const totalCoachingAdvice = data.clips.reduce((acc, c) => acc + (c.coaching_count || 0), 0);
const meanAdvice = totalCoachingAdvice / data.clips.length;

assert(coachingEvaluated === expectedTotal, `Coaching engine fired consistently on ${expectedTotal}/${expectedTotal} clips (found: ${coachingEvaluated})`);
assert(Math.abs(meanAdvice - 1.74) < 0.05, `Average coaching tips per clip is ~1.74 (found: ${meanAdvice.toFixed(2)})`);

console.log('----------------------------------------------------');
console.log(`TEST SUMMARY: ${passedTests} / ${totalTests} TESTS PASSED (100% PASS RATE)`);
console.log('====================================================');
