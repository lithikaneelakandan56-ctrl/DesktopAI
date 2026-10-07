# audio_processor.py
"""Audio processing functionality separated from the main assistant."""

import os

import numpy as np
import soundfile as sf
from groq import Groq

from src.config import config
from src.utils.logger import get_logger

logger = get_logger()


class AudioProcessor:
    """Handles audio transcription and processing."""

    def __init__(self):
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)

    def process_audio(self, audio_data):
        """Process audio data and return transcription."""
        logger.info("Processing audio...")
        temp_file = config.TEMP_AUDIO_FILE

        try:
            # Convert audio bytes to numpy array and save as WAV
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sf.write(temp_file, audio_array, config.SAMPLE_RATE)

            with open(temp_file, "rb") as f:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=("temp.wav", f.read()),
                    model=config.TRANSCRIPTION_MODEL,
                    response_format="text",
                    prompt=config.TRANSCRIPTION_PROMPT,
                )
                logger.info(f"Transcription: {transcription}")
                return transcription

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
