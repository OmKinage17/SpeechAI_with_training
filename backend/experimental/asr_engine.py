"""
ASR Engine Module for Experimental SpeechAI Pipeline (Process-Isolated)
======================================================================
Provides safe, process-isolated Whisper inference with pre-emptive OS-level
process termination on timeouts, preventing lingering threads or queue backlogs.
Compatible with Windows multiprocessing spawn semantics.
"""

import os
import time
import logging
import multiprocessing
from typing import Dict, Any, Optional

from .config import ASRConfig
from .asr_worker import worker_process_main

logger = logging.getLogger("SpeechAI_Experimental_ASR")

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
                    return {
                        "text": res.get("text", ""),
                        "transcript": res.get("transcript", ""),
                        "segments": res.get("segments", []),
                        "words": res.get("words", []),
                        "words_list": res.get("words_list", []),
                        "latency_sec": dt,
                        "status": "SUCCESS"
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
