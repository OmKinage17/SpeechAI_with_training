"""
ASR Engine Module for Experimental SpeechAI Pipeline (Process-Isolated)
======================================================================
Provides safe, process-isolated Whisper inference with pre-emptive OS-level
process termination on timeouts, preventing lingering threads or queue backlogs.
Compatible with Windows multiprocessing spawn semantics.
"""

import os
import re
import time
import logging
import multiprocessing
from typing import Dict, Any, Optional, List, Tuple

from .config import ASRConfig
from .asr_worker import worker_process_main

logger = logging.getLogger("SpeechAI_Experimental_ASR")

HESITATION_WORDS = {'um', 'uh', 'er', 'ah', 'hmm', 'umm', 'uhh', 'erm'}


def detect_and_sanitize_repetition_loop(
    raw_transcript: str,
    raw_words_list: Optional[List[Dict[str, Any]]] = None,
    clip_duration_sec: float = 3.0
) -> Dict[str, Any]:
    """
    Detects and sanitizes degenerate Whisper repetition loops without suppressing
    genuine clinically-relevant speech disfluencies.
    
    Preserves:
    - Normal utterances (< 12 words) completely untouched.
    - Genuine stuttering, lexical repetitions, and prolongations (e.g. 'did did', 'um uh').
    - Separate storage of raw_transcript vs sanitized_transcript.
    
    Identifies:
    1. Runaway hesitation streams (>= 6 consecutive fillers with token explosion >= 25 words).
    2. Runaway single-token loops (>= 5 consecutive repeats with token explosion >= 25 words, or >= 8 repeats).
    3. Runaway alternating 2-gram loops (>= 3 cycles with token explosion >= 25 words, or >= 5 cycles).
    """
    words = raw_transcript.split()
    total_words = len(words)
    raw_wl = raw_words_list or []

    # Safeguard 1: Normal short utterance - clinically genuine disfluencies preserved
    if total_words < 12:
        return {
            'is_loop': False,
            'status': 'SUCCESS',
            'reason': 'Normal token count (< 12 words); genuine disfluencies preserved.',
            'raw_transcript': raw_transcript,
            'sanitized_transcript': raw_transcript,
            'sanitized_words_list': raw_wl,
            'loop_pattern': None,
            'truncated_tokens_count': 0
        }

    clean_tokens = [re.sub(r'[^\w]', '', w).lower() for w in words]
    first_loop_idx = None
    loop_reason = ""
    loop_pattern = {}

    # Check 1: Continuous runaway hesitation/filler loop (>= 6 consecutive hesitations)
    hesitation_streak_start = None
    hesitation_streak_len = 0
    for i, t in enumerate(clean_tokens):
        if t in HESITATION_WORDS:
            if hesitation_streak_start is None:
                hesitation_streak_start = i
            hesitation_streak_len += 1
            if hesitation_streak_len >= 6 and total_words >= 25:
                # Keep initial 3 hesitations to preserve dysfluency evidence, truncate runaway tail
                keep_len = hesitation_streak_start + 3
                first_loop_idx = keep_len
                loop_reason = (
                    f"Degenerate hesitation stream: {hesitation_streak_len}+ consecutive filler tokens. "
                    f"Truncated {total_words - keep_len} runaway tokens while preserving initial hesitations."
                )
                loop_pattern = {'type': 'hesitation_stream', 'start': hesitation_streak_start, 'streak_len': hesitation_streak_len}
                break
        else:
            hesitation_streak_start = None
            hesitation_streak_len = 0

    # Check 2: Single lexical token runaway (e.g. 'did did did did did' >= 5 repetitions when total_words >= 25)
    if first_loop_idx is None:
        for i in range(len(clean_tokens) - 4):
            t = clean_tokens[i]
            if not t:
                continue
            k = 1
            while (i + k) < len(clean_tokens) and clean_tokens[i + k] == t:
                k += 1
            if (k >= 5 and total_words >= 25) or k >= 8:
                keep_len = i + 2  # Keep 2 instances of repeated token to preserve stutter evidence
                first_loop_idx = keep_len
                loop_reason = (
                    f"Degenerate single-token loop: '{t}' repeated {k} times consecutively. "
                    f"Truncated {total_words - keep_len} runaway tokens while preserving initial repetition."
                )
                loop_pattern = {'type': 'single_token', 'token': t, 'count': k}
                break

    # Check 3: Alternating 2-gram runaway (>= 3 cycles = 6 tokens when total_words >= 25)
    if first_loop_idx is None:
        for i in range(len(clean_tokens) - 5):
            t1, t2 = clean_tokens[i], clean_tokens[i+1]
            if t1 and t2 and t1 != t2:
                cycles = 1
                while (i + (cycles + 1) * 2) <= len(clean_tokens):
                    if clean_tokens[i + cycles * 2] == t1 and clean_tokens[i + cycles * 2 + 1] == t2:
                        cycles += 1
                    else:
                        break
                if (cycles >= 3 and total_words >= 25) or cycles >= 5:
                    keep_len = i + 4  # Keep 2 cycles
                    first_loop_idx = keep_len
                    loop_reason = (
                        f"Degenerate 2-gram loop: '{t1} {t2}' repeated {cycles} cycles. "
                        f"Truncated {total_words - keep_len} runaway tokens while preserving initial repetition."
                    )
                    loop_pattern = {'type': '2gram', 'pattern': [t1, t2], 'cycles': cycles}
                    break

    if first_loop_idx is not None and first_loop_idx < total_words:
        sanitized_words = words[:first_loop_idx]
        sanitized_text = ' '.join(sanitized_words)
        truncated_count = total_words - first_loop_idx
        sanitized_wl = raw_wl[:first_loop_idx] if raw_wl else []
        return {
            'is_loop': True,
            'status': 'DEGENERATE_LOOP_RECOVERED',
            'reason': loop_reason,
            'raw_transcript': raw_transcript,
            'sanitized_transcript': sanitized_text,
            'sanitized_words_list': sanitized_wl,
            'loop_pattern': loop_pattern,
            'truncated_tokens_count': truncated_count
        }

    return {
        'is_loop': False,
        'status': 'SUCCESS',
        'reason': 'No degenerate repetition loop detected; genuine disfluencies preserved.',
        'raw_transcript': raw_transcript,
        'sanitized_transcript': raw_transcript,
        'sanitized_words_list': raw_wl,
        'loop_pattern': None,
        'truncated_tokens_count': 0
    }


class ExperimentalASREngine:
    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self._process = None
        self._conn = None

    def _ensure_worker(self) -> bool:
        """
        Ensures a healthy child worker process is running and ready.
        If the previous worker timed out or crashed, spawns a clean replacement.
        """
        if self._process is not None and self._process.is_alive() and self._conn is not None:
            return True

        self._terminate_worker()

        try:
            parent_conn, child_conn = multiprocessing.Pipe()
            self._conn = parent_conn

            self._process = multiprocessing.Process(
                target=worker_process_main,
                args=(child_conn, self.config.model_name, self.config.initial_prompt),
                daemon=True
            )
            self._process.start()
            child_conn.close()

            # Wait for worker initialization (loading Whisper model weights)
            init_timeout = 30.0
            if self._conn.poll(timeout=init_timeout):
                msg = self._conn.recv()
                if msg.get("status") == "READY":
                    logger.info(f"ASR child worker (PID {self._process.pid}) initialized and ready.")
                    return True
                else:
                    logger.error(f"ASR child worker initialization failed: {msg}")
                    self._terminate_worker()
                    return False
            else:
                logger.error(f"ASR child worker initialization timed out after {init_timeout}s.")
                self._terminate_worker()
                return False
        except Exception as e:
            logger.error(f"Failed to spawn ASR child worker: {e}")
            self._terminate_worker()
            return False

    def _terminate_worker(self):
        """Forcefully terminates and reaps the child worker process and closes IPC."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._process is not None:
            pid = self._process.pid
            try:
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
                    if self._process.is_alive():
                        logger.warning(f"Process {pid} did not terminate gracefully; issuing kill.")
                        self._process.kill()
                        self._process.join(timeout=1.0)
            except Exception as e:
                logger.warning(f"Error terminating worker process {pid}: {e}")
            finally:
                self._process = None

    def transcribe(
        self,
        audio_path: str,
        timeout_sec: Optional[float] = None,
        bounded_temperature: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribes audio with process-level isolation and strict OS termination on timeout.
        """
        if not os.path.exists(audio_path):
            return {
                "text": "",
                "transcript": "",
                "segments": [],
                "words": [],
                "words_list": [],
                "latency_sec": 0.0,
                "status": "ERROR",
                "error": f"File not found: {audio_path}"
            }

        timeout = timeout_sec or self.config.per_clip_timeout_sec
        t0 = time.time()

        if not self._ensure_worker():
            dt = round(time.time() - t0, 3)
            return {
                "text": "",
                "transcript": "",
                "segments": [],
                "words": [],
                "words_list": [],
                "latency_sec": dt,
                "status": "ERROR",
                "error": "Failed to spawn or initialize ASR child worker process"
            }

        temp_setting = self.config.temperature if bounded_temperature else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        req_msg = {
            "audio_path": audio_path,
            "temperature": temp_setting,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "language": self.config.language,
            "word_timestamps": self.config.word_timestamps,
            "beam_size": self.config.beam_size,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "logprob_threshold": self.config.logprob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold
        }

        try:
            self._conn.send(req_msg)
            
            # Wait with hard timeout
            if self._conn.poll(timeout=timeout):
                res = self._conn.recv()
                dt = round(time.time() - t0, 3)
                
                if not isinstance(res, dict):
                    self._terminate_worker()
                    return {
                        "text": "",
                        "transcript": "",
                        "segments": [],
                        "words": [],
                        "words_list": [],
                        "latency_sec": dt,
                        "status": "ERROR",
                        "error": "Serialization or IPC error: response is not a valid dictionary"
                    }

                st = res.get("status", "SUCCESS")
                if st == "SUCCESS":
                    raw_text = res.get("text", "")
                    raw_words = res.get("words_list", [])
                    
                    # Sanitize repetition loops while preserving raw transcripts and genuine disfluencies
                    loop_res = detect_and_sanitize_repetition_loop(
                        raw_transcript=raw_text,
                        raw_words_list=raw_words,
                        clip_duration_sec=3.0
                    )

                    return {
                        "text": loop_res["sanitized_transcript"],
                        "transcript": loop_res["sanitized_transcript"],
                        "raw_text": raw_text,
                        "raw_transcript": raw_text,
                        "segments": res.get("segments", []),
                        "words": loop_res["sanitized_words_list"],
                        "words_list": loop_res["sanitized_words_list"],
                        "raw_words": raw_words,
                        "raw_words_list": raw_words,
                        "latency_sec": dt,
                        "status": loop_res["status"],
                        "is_loop_detected": loop_res["is_loop"],
                        "loop_pattern": loop_res["loop_pattern"],
                        "sanitization_reason": loop_res["reason"],
                        "truncated_tokens_count": loop_res["truncated_tokens_count"]
                    }
                else:
                    return {
                        "text": "",
                        "transcript": "",
                        "segments": [],
                        "words": [],
                        "words_list": [],
                        "latency_sec": dt,
                        "status": "ERROR",
                        "error": res.get("error", "Unknown worker error")
                    }
            else:
                # TIMEOUT EXPIRED: Terminate OS worker process immediately
                dt = round(time.time() - t0, 3)
                logger.warning(
                    f"ASR worker (PID {self._process.pid if self._process else 'N/A'}) timed out on "
                    f"'{os.path.basename(audio_path)}' after {dt}s (limit: {timeout}s). Terminating process."
                )
                self._terminate_worker()
                return {
                    "text": "",
                    "transcript": "",
                    "segments": [],
                    "words": [],
                    "words_list": [],
                    "latency_sec": dt,
                    "status": "TIMEOUT",
                    "error": f"Worker process timed out after {timeout}s and was terminated"
                }
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            dt = round(time.time() - t0, 3)
            logger.error(f"ASR worker process crashed unexpectedly on '{os.path.basename(audio_path)}': {e}")
            self._terminate_worker()
            return {
                "text": "",
                "transcript": "",
                "segments": [],
                "words": [],
                "words_list": [],
                "latency_sec": dt,
                "status": "CRASHED",
                "error": f"Worker process terminated unexpectedly: {e}"
            }
        except Exception as e:
            dt = round(time.time() - t0, 3)
            logger.error(f"Unexpected error communicating with ASR worker: {e}")
            self._terminate_worker()
            return {
                "text": "",
                "transcript": "",
                "segments": [],
                "words": [],
                "words_list": [],
                "latency_sec": dt,
                "status": "ERROR",
                "error": str(e)
            }

    # Method alias for compatibility
    transcribe_audio_safe = transcribe

    def shutdown(self):
        """Gracefully shuts down the ASR child process."""
        if self._conn is not None and self._process is not None and self._process.is_alive():
            try:
                self._conn.send(None)
                self._process.join(timeout=1.5)
            except Exception:
                pass
        self._terminate_worker()

    def __del__(self):
        self.shutdown()
