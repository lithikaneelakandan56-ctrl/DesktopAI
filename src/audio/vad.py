"""
Voice Activity Detector (VAD) — Standalone module.

NOTE: This module is NOT currently used by the main application.
The Listener class in `listener.py` handles wake word detection, speaker
verification, and voice activity detection together. This standalone
VoiceActivityDetector class is kept as a reference implementation for
potential future use — e.g., if we want to decouple VAD into its own
component or use it in a different pipeline.
"""

import collections
import os
import struct
import sys
import threading
import time

import numpy as np
import pvporcupine
import sounddevice as sd
import soundfile as sf
import torch
from resemblyzer import VoiceEncoder, preprocess_wav

torch.set_num_threads(1)

from src.config import config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.thread_executor import executor  # noqa: E402

model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")

get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils

logger = get_logger()


class VoiceActivityDetector:
    def __init__(self, tts_player=None, overlay=None):
        # Audio config
        self.SAMPLE_RATE = config.SAMPLE_RATE
        self.CHANNELS = config.CHANNELS
        self.CHUNK = config.CHUNK_SIZE
        self.num_samples = config.NUM_SAMPLES
        self.recording = False
        self.listening = False
        self.lock = threading.Lock()
        self.tts_player = tts_player
        self.overlay = overlay

        # Load sound effects
        self.start_sound, self.start_sr = sf.read(config.START_SOUND_FILE)
        self.stop_sound, self.stop_sr = sf.read(config.STOP_SOUND_FILE)

        # Initialize Porcupine
        self.porcupine = pvporcupine.create(
            access_key=config.PORCUPINE_ACCESS_KEY,
            keyword_paths=[config.WAKE_WORD_MODEL],
            sensitivities=[0.95],
        )

        # Initialize audio stream
        self.stream = sd.InputStream(
            channels=self.CHANNELS,
            samplerate=self.SAMPLE_RATE,
            dtype="int16",
            blocksize=self.num_samples,
        )
        self.stream.start()

        # Initialize audio output stream
        self.output_stream = sd.OutputStream(samplerate=44100, channels=1, dtype="float32")
        self.output_stream.start()

        self.confidence_threshold = config.CONFIDENCE_THRESHOLD

        self.voice_encoder = VoiceEncoder()
        self.OWNER_VOICE_FILE = config.OWNER_VOICE_FILE
        self.owner_embeddings = None

        if os.path.exists(self.OWNER_VOICE_FILE):
            self.owner_embeddings = self.voice_encoder.embed_utterance(preprocess_wav(self.OWNER_VOICE_FILE))
            logger.info("✅ Owner voice file found. Loading embeddings...")
        else:
            logger.error(
                "Owner voice file not found. Please record your voice and save it in data/owner.wav\n"
                "Run `python setup/record_owner_voice.py` to record your voice."
            )
            sys.exit(1)

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype("float32")
        if abs_max > 0:
            sound *= 1 / 32768
        return sound.squeeze()

    def record_audio(self, frames=None):
        if frames is None:
            frames = []
        with self.lock:
            if self.overlay:
                self.overlay.put_message("status", "Listening...", "skyblue")
            self.recording = True
            logger.debug("Recording started...")
            # Play start sound effect
            self.play_sound(self.start_sound, self.start_sr)

            silence_frames = 0
            max_silence_frames = config.MAX_SILENCE_FRAMES
            max_recording_frames = config.MAX_RECORDING_FRAMES

            while silence_frames < max_silence_frames and len(frames) < max_recording_frames:
                audio_chunk, _ = self.stream.read(self.num_samples)
                audio_float32 = self.int2float(audio_chunk)
                frames.append(audio_chunk)

                confidence = model(torch.from_numpy(audio_float32), self.SAMPLE_RATE).item()

                if confidence > self.confidence_threshold:
                    silence_frames = 0
                else:
                    silence_frames += 1

            # Convert frames to numpy array
            audio_data = np.concatenate(frames[:-60])
            self.recording = False
            logger.debug("Recording stopped...")
            # Play stop sound effect
            self.play_sound(self.stop_sound, self.stop_sr)

            if self.overlay:
                self.overlay.put_message("status", "Active", "green")
            return audio_data

    def listen(self, func):
        self.listening = True
        frame_queue = collections.deque(maxlen=60)  # for audio_chunk frames (int16)
        float_queue = collections.deque(maxlen=60)  # for speaker verification (float32)

        while self.listening:
            if self.recording:
                time.sleep(0.01)  # Add small delay to prevent busy waiting
                continue

            audio_chunk, _ = self.stream.read(self.num_samples)
            audio_float = self.int2float(audio_chunk)

            # Add current frame to queue
            frame_queue.append(audio_chunk)
            float_queue.append(audio_float)

            # Check for wake word if not already recording
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, audio_chunk.tobytes())
            keyword_index = self.porcupine.process(pcm)
            if keyword_index == -1:
                continue
            logger.debug("🎯 Wake word detected!")

            # === Speaker verification before continuing ===
            if self.owner_embeddings is not None:
                float_buffer = np.concatenate(list(float_queue))  # ~1s of audio
                predicted_embedding = self.voice_encoder.embed_utterance(float_buffer)
                similarity = np.dot(self.owner_embeddings, predicted_embedding)

                logger.debug(f"🔍 Speaker similarity: {similarity:.3f}")
                if similarity < config.SPEAKER_SIMILARITY_THRESHOLD:
                    logger.warning("🚫 Voice does not match owner. Ignoring wake word.")
                    continue  # Skip further processing

                logger.info("✅ Speaker verified as owner")

            # Stop TTS playback if available
            if self.tts_player:
                self.tts_player.stop_current()
                logger.debug("Stopped TTS playback due to wake word detection")

            audio_data = self.record_audio(list(frame_queue))

            # Process in a separate thread to avoid blocking the main loop
            future = executor.submit(func, audio_data)
            try:
                future.result(timeout=config.AUDIO_PROCESSING_TIMEOUT)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error processing audio: {e}")

        logger.info("👋 Stopped listening")

    def stop_listening(self):
        self.listening = False
        if hasattr(self, "porcupine"):
            self.porcupine.delete()
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, "output_stream"):
            self.output_stream.stop()
            self.output_stream.close()

    def play_sound(self, sound_data, sample_rate):
        # Stop any currently playing sound
        self.output_stream.stop()
        self.output_stream.start()
        # Convert sound data to float32 before playing
        sound_data = sound_data.astype("float32")
        # Play the new sound
        self.output_stream.write(sound_data)

    def play_audio(self, audio_data):
        duration_seconds = len(audio_data) / self.SAMPLE_RATE
        logger.debug(f"Received audio data of length: {len(audio_data)} samples ({duration_seconds:.2f} seconds)")

        # Play back the recorded audio
        sd.play(audio_data, self.SAMPLE_RATE)
        sd.wait()  # Wait until audio is finished playing


if __name__ == "__main__":
    # Example usage
    vad = VoiceActivityDetector()
    vad.listen(vad.play_audio)
