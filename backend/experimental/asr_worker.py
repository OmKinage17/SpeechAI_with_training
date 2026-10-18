"""
Isolated Worker Process Module for Windows-Compatible Whisper ASR
================================================================
Runs in a separate OS process, loading Whisper and communicating via Pipe.
Enables true OS-level process termination on timeouts.
"""

import os
import sys
import time
import logging

def worker_process_main(conn, model_name: str, initial_prompt: str):
    """
    Child process entrypoint for ASR inference.
    Listens for transcription requests on conn until sentinel None is received.
    """
    try:
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
        except Exception:
            pass

        import whisper
        # Load model inside child process
        model = whisper.load_model(model_name)
        # Notify parent that worker is ready
        conn.send({"status": "READY"})
    except Exception as e:
        try:
            conn.send({"status": "INIT_ERROR", "error": str(e)})
        except Exception:
            pass
        return

    # Request processing loop
    while True:
        try:
            if not conn.poll(timeout=None):
                continue
            msg = conn.recv()
            if msg is None:  # Shutdown sentinel
                break

            audio_path = msg.get("audio_path")
            temperature = msg.get("temperature", (0.0, 0.2))
            language = msg.get("language", "en")
            condition_on_previous_text = msg.get("condition_on_previous_text", False)
            word_timestamps = msg.get("word_timestamps", True)
            beam_size = msg.get("beam_size", 5)
            compression_ratio_threshold = msg.get("compression_ratio_threshold", 2.4)
            logprob_threshold = msg.get("logprob_threshold", -1.0)
            no_speech_threshold = msg.get("no_speech_threshold", 0.6)

            t0 = time.time()
            result = model.transcribe(
                audio_path,
                temperature=temperature,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=word_timestamps,
                language=language,
                beam_size=beam_size,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold
            )
            dt = round(time.time() - t0, 3)

            transcript = result.get("text", "").strip()
            segments = result.get("segments", [])
            words_list = []
            for seg in segments:
                if "words" in seg:
                    words_list.extend(seg["words"])

            conn.send({
                "status": "SUCCESS",
                "transcript": transcript,
                "text": transcript,
                "segments": segments,
                "words": words_list,
                "words_list": words_list,
                "inference_time_sec": dt
            })
        except Exception as e:
            try:
                conn.send({
                    "status": "ERROR",
                    "error": str(e),
                    "transcript": "",
                    "text": "",
                    "segments": [],
                    "words": [],
                    "words_list": []
                })
            except Exception:
                break

    try:
        conn.close()
    except Exception:
        pass
