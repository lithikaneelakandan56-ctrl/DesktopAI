import os
import sys
import time

import sounddevice as sd
import soundfile as sf

from src.config import config
from src.utils.logger import get_logger

SAMPLE_RATE = config.SAMPLE_RATE
DURATION = 20  # seconds
OWNER_FILE = config.OWNER_VOICE_FILE

logger = get_logger()


def main():
    try:
        print(f"""
        You'll need to speak for {DURATION} seconds to register your voice.
        Please find a quiet place to ensure the best recording quality.
        """)
        input("Press Enter to start recording...")

        print(f"Recording for {DURATION} seconds...")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )

        start_time = time.time()
        while int(time.time() - start_time) < DURATION:
            remaining = DURATION - int(time.time() - start_time)
            print(f"\rRecording... {remaining} seconds remaining ", end="", flush=True)
            time.sleep(1)

        sd.wait()
        audio = audio.flatten()

        # Ensure the data directory exists
        os.makedirs(os.path.dirname(OWNER_FILE), exist_ok=True)

        sf.write(OWNER_FILE, audio, SAMPLE_RATE)
        print()  # Newline after the countdown
        logger.info(f"Voice profile saved to {OWNER_FILE}")
        print(f"✅ Voice profile saved to {OWNER_FILE}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during recording: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
