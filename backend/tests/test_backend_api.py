import asyncio
import requests
import wave
import struct
import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import main

def generate_silent_wav(path, duration=2.0, sample_rate=16000):
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        num_samples = int(duration * sample_rate)
        data = struct.pack(f'<{num_samples}h', *([0] * num_samples))
        wav_file.writeframes(data)

def test_analyze_endpoint():
    print("\nTesting /analyze/speech endpoint...")
    temp_audio = "temp_analyze_test.wav"
    generate_silent_wav(temp_audio)

    url = "http://127.0.0.1:8000/analyze/speech"
    data = {"user_id": "test_user_001"}
    try:
        with open(temp_audio, "rb") as f:
            files = {"audio": (temp_audio, f, "audio/wav")}
            response = requests.post(url, files=files, data=data)
        print("Status Code:", response.status_code)
        if response.status_code == 200:
            print("Response JSON:")
            import json
            print(json.dumps(response.json(), indent=2))
        else:
            print("Response Text:", response.text)
    except Exception as e:
        print("Error during request:", e)
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

def test_tts_audio_is_buffered_before_playback():
    async def fake_stream():
        yield {"type": "audio", "data": b"abc"}
        yield {"type": "audio", "data": b"def"}
        yield {"type": "audio", "data": b"ghi"}

    result = asyncio.run(main.collect_tts_audio(fake_stream()))
    assert result == b"abcdefghi"

if __name__ == "__main__":
    test_analyze_endpoint()
    test_tts_audio_is_buffered_before_playback()
