"""
Acoustic Feature Extractor Module for SpeechAI Experimental Pipeline
====================================================================
Uses librosa, numpy, and scipy to compute acoustic features:
- Voicing and pitch stability (F0)
- Spectral flux and centroid (articulatory posture stability)
- Energy profile (RMS, noise floor, silence boundaries)
- Formant energy concentration (schwa / nasal resonance verification)
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf

from .config import AcousticConfig

logger = logging.getLogger("SpeechAI_Acoustic_Features")

class AcousticFeatureExtractor:
    def __init__(self, config: Optional[AcousticConfig] = None):
        self.config = config or AcousticConfig()

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Loads audio file and resamples to target sample rate (default 16kHz)."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        return y, sr

    def compute_energy_and_noise_floor(self, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Computes frame-level RMS energy and estimates the background noise floor.
        Returns: (rms_frames, noise_floor, vocal_threshold)
        """
        frame_len = int(self.config.sample_rate * (self.config.frame_length_ms / 1000.0))
        hop_len = int(self.config.sample_rate * (self.config.hop_length_ms / 1000.0))
        
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        
        if len(rms) == 0:
            return np.array([0.0]), 1e-4, 1e-3

        # Noise floor estimated from the quietest percentile of frames
        pct = self.config.noise_floor_percentile
        cutoff_idx = max(1, int(len(rms) * (pct / 100.0)))
        sorted_rms = np.sort(rms)
        noise_floor = float(np.median(sorted_rms[:cutoff_idx]))
        noise_floor = max(0.003, min(0.04, noise_floor))

        vocal_threshold = float(noise_floor * self.config.vocal_energy_multiplier)
        return rms, noise_floor, vocal_threshold

    def extract_pitch_and_voicing(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts fundamental frequency (F0) using pyin.
        Returns: (f0, voiced_flag, voiced_probs)
        """
        hop_len = int(self.config.sample_rate * (self.config.hop_length_ms / 1000.0))
        frame_len = int(self.config.sample_rate * (self.config.frame_length_ms / 1000.0))
        
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=self.config.f0_min_hz,
                fmax=self.config.f0_max_hz,
                sr=self.config.sample_rate,
                frame_length=frame_len,
                hop_length=hop_len
            )
            return f0, voiced_flag, voiced_probs
        except Exception as e:
            logger.warning(f"pYIN pitch tracking error: {e}, falling back to zero pitch")
            n_frames = 1 + len(y) // hop_len
            return np.zeros(n_frames), np.zeros(n_frames, dtype=bool), np.zeros(n_frames)

    def compute_spectral_stability(self, y: np.ndarray) -> np.ndarray:
        """
        Computes spectral flux (difference between consecutive power spectra).
        Low spectral flux during voiced frames indicates a steady, held articulatory posture (prolongation).
        """
        hop_len = int(self.config.sample_rate * (self.config.hop_length_ms / 1000.0))
        S = np.abs(librosa.stft(y, hop_length=hop_len))
        
        # Spectral flux: Euclidean distance between normalized spectra
        S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
        flux = np.sqrt(np.sum(np.diff(S_norm, axis=1)**2, axis=0))
        # Pad beginning to match frame count
        flux = np.pad(flux, (1, 0), mode='edge')
        return flux

    def find_prolongation_candidates(
        self, 
        y: np.ndarray, 
        min_duration_sec: float = 0.50,
        max_pitch_std_st: float = 1.2,
        max_spectral_flux: float = 0.20
    ) -> List[Dict[str, Any]]:
        """
        Scans audio for sustained vowels/voiced consonants exceeding min_duration_sec
        where pitch variation is very low (< 1.2 semitones) and spectral flux is low (< 0.20).
        """
        hop_len = int(self.config.sample_rate * (self.config.hop_length_ms / 1000.0))
        frame_sec = self.config.hop_length_ms / 1000.0

        f0, voiced, _ = self.extract_pitch_and_voicing(y)
        rms, _, vocal_thresh = self.compute_energy_and_noise_floor(y)
        flux = self.compute_spectral_stability(y)

        candidates = []
        in_voiced = False
        start_frame = 0

        min_frames = int(min_duration_sec / frame_sec)

        def check_segment(s_idx: int, e_idx: int):
            dur_frames = e_idx - s_idx
            if dur_frames >= min_frames:
                seg_f0 = f0[s_idx:e_idx]
                valid_f0 = seg_f0[~np.isnan(seg_f0)]
                if len(valid_f0) >= 0.5 * dur_frames:
                    st = 12.0 * np.log2(valid_f0 / (np.median(valid_f0) + 1e-6))
                    pitch_std_st = float(np.std(st))
                    seg_flux = flux[s_idx:e_idx]
                    mean_flux = float(np.mean(seg_flux)) if len(seg_flux) > 0 else 0.0

                    if pitch_std_st < max_pitch_std_st and mean_flux < max_spectral_flux:
                        candidates.append({
                            "start": round(s_idx * frame_sec, 2),
                            "end": round(e_idx * frame_sec, 2),
                            "duration": round(dur_frames * frame_sec, 2),
                            "pitch_std_st": round(pitch_std_st, 2),
                            "mean_flux": round(mean_flux, 4),
                            "confidence": round(min(1.0, (dur_frames * frame_sec) / 0.8), 2)
                        })

        for i in range(len(voiced)):
            is_active = voiced[i] and (i < len(rms) and rms[i] >= vocal_thresh * 0.5)
            
            if is_active and not in_voiced:
                in_voiced = True
                start_frame = i
            elif not is_active and in_voiced:
                in_voiced = False
                check_segment(start_frame, i)

        if in_voiced:
            check_segment(start_frame, len(voiced))

        return candidates

    def compute_harmonic_voicing(self, segment: np.ndarray) -> float:
        """
        Computes the proportion of frames in a segment that exhibit periodic harmonic
        vocal fold vibration (voicing) in the human pitch range (f0_min to f0_max).
        Uses normalized frame autocorrelation with local peak detection.
        Returns:
            voicing_ratio: float between 0.0 and 1.0 (voiced_frames / total_frames)
        """
        sr = self.config.sample_rate
        frame_len = int(sr * (self.config.frame_length_ms / 1000.0))
        hop_len = int(sr * (self.config.hop_length_ms / 1000.0))
        min_lag = max(1, int(sr / self.config.f0_max_hz))
        max_lag = min(frame_len - 1, int(sr / self.config.f0_min_hz))
        
        if len(segment) < frame_len or max_lag <= min_lag:
            return 0.0
            
        voiced_frames = 0
        total_frames = 0
        min_autocorr = self.config.autocorr_voicing_threshold
        
        for start in range(0, len(segment) - frame_len + 1, hop_len):
            frame = segment[start:start+frame_len]
            frame = frame - np.mean(frame)
            energy = np.sum(frame**2)
            if energy > 1e-7:
                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(frame)-1:]
                norm = corr[0]
                if norm > 1e-8:
                    search_region = corr[min_lag:max_lag] / norm
                    # Look for prominent local peak in valid pitch lag range
                    for idx in range(1, len(search_region) - 1):
                        val = search_region[idx]
                        if val >= min_autocorr and val > search_region[idx - 1] and val >= search_region[idx + 1]:
                            voiced_frames += 1
                            break
            total_frames += 1
            
        return float(voiced_frames) / max(1, total_frames)

    def verify_filled_pause_resonance(
        self, 
        y: np.ndarray, 
        start_sec: float, 
        end_sec: float,
        vocal_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a candidate pause interval to determine if it contains vocalic /ə/ or /m/
        formant resonance vs unvoiced noise (breathing, clicks, room tone, silence).
        
        Fix 1 enhancements:
        - Validates segment RMS energy against absolute floor and clip vocal threshold
        - Validates harmonic voicing evidence (periodic pitch correlation)
        - Retains zero-crossing rate and formant/rumble spectral balance
        """
        sr = self.config.sample_rate
        s_idx = max(0, int(start_sec * sr))
        e_idx = min(len(y), int(end_sec * sr))
        
        if e_idx - s_idx < int(0.15 * sr):
            return {"is_filled_pause": False, "reason": "too_short", "schwa_energy_ratio": 0.0}

        segment = y[s_idx:e_idx]
        
        # 1. RMS Energy Floor Check (rejects studio silence and quiet room tone)
        seg_rms = float(np.sqrt(np.mean(segment**2)))
        energy_floor = max(
            self.config.min_filled_pause_rms,
            (vocal_threshold * self.config.vocal_energy_floor_multiplier) if vocal_threshold is not None else 0.0
        )
        if seg_rms < energy_floor:
            return {
                "is_filled_pause": False,
                "reason": "below_energy_floor",
                "seg_rms": round(seg_rms, 5),
                "energy_floor": round(energy_floor, 5),
                "schwa_energy_ratio": 0.0,
                "voicing_ratio": 0.0
            }

        # 2. Zero crossing rate (unvoiced breath noise has high ZCR > 0.35)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(segment)[0]))
        if zcr > 0.35:
            return {
                "is_filled_pause": False,
                "reason": "high_zcr_unvoiced",
                "zcr": round(zcr, 3),
                "seg_rms": round(seg_rms, 5),
                "schwa_energy_ratio": 0.0
            }

        # 3. Voicing Evidence Check (real vocalic hesitation has periodic pitch in 75-400 Hz)
        voicing_ratio = self.compute_harmonic_voicing(segment)
        if voicing_ratio < self.config.min_filled_pause_voicing_ratio:
            return {
                "is_filled_pause": False,
                "reason": "unvoiced_insufficient_pitch",
                "voicing_ratio": round(voicing_ratio, 3),
                "seg_rms": round(seg_rms, 5),
                "zcr": round(zcr, 3),
                "schwa_energy_ratio": 0.0
            }

        # 4. Power spectrum analysis
        S = np.abs(np.fft.rfft(segment))**2
        freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)
        total_energy = float(np.sum(S)) + 1e-9

        # Central vowel / schwa / nasal frequency band: 300 - 1800 Hz
        schwa_mask = (freqs >= 300.0) & (freqs <= 1800.0)
        schwa_energy = float(np.sum(S[schwa_mask]))
        schwa_ratio = schwa_energy / total_energy

        # Low frequency breath rumble (< 100 Hz)
        low_rumble_mask = freqs < 100.0
        rumble_ratio = float(np.sum(S[low_rumble_mask])) / total_energy

        # Real filled pause (uh/um) has high energy in central formant band, low ZCR, low rumble
        is_filled = (schwa_ratio >= self.config.min_formant_schwa_ratio) and (rumble_ratio < 0.40) and (zcr < 0.25)
        reason = "verified_vocalic_resonance" if is_filled else (
            "insufficient_schwa_ratio" if schwa_ratio < self.config.min_formant_schwa_ratio else (
                "excessive_low_rumble" if rumble_ratio >= 0.40 else "elevated_zcr"
            )
        )

        return {
            "is_filled_pause": is_filled,
            "reason": reason,
            "schwa_energy_ratio": round(schwa_ratio, 3),
            "rumble_ratio": round(rumble_ratio, 3),
            "zcr": round(zcr, 3),
            "seg_rms": round(seg_rms, 5),
            "voicing_ratio": round(voicing_ratio, 3)
        }
