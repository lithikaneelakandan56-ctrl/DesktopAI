import asyncio
import os
import queue
import tempfile
import threading

import edge_tts
import pyaudio
from pydub import AudioSegment

from src.config import config
from src.utils.logger import get_logger

logger = get_logger()


def _write_audio_chunk(filename, data):
    with open(filename, "ab") as f:
        f.write(data)


class TTSPlayer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.current_playback_event = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        while not self.stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text:  # Make sure we have text to speak
                    self._speak(text)
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS thread: {e}")

    async def synthesize_to_file(self, text: str, filename: str):
        try:
            communicate = edge_tts.Communicate(text=text, voice=config.TTS_VOICE, rate=config.TTS_RATE)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    await asyncio.to_thread(
                        _write_audio_chunk,
                        filename,
                        chunk["data"],
                    )
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            raise

    def stop_current(self):
        logger.debug("Stopping current speech playback")
        self.current_playback_event.set()
        # Small delay to ensure playback stops before new speech begins
        threading.Event().wait(0.1)
        self.current_playback_event.clear()

    def _play_audio_segment(self, audio_segment):
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(audio_segment.sample_width),
                channels=audio_segment.channels,
                rate=audio_segment.frame_rate,
                output=True,
            )

            chunk_size = config.TTS_CHUNK_SIZE
            audio_data = audio_segment.raw_data

            for i in range(0, len(audio_data), chunk_size):
                if self.stop_event.is_set() or self.current_playback_event.is_set():
                    break
                stream.write(audio_data[i : i + chunk_size])
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()

    def _speak(self, text):
        if not text or not text.strip():
            logger.warning("Empty text passed to speech engine")
            return

        logger.debug(f"🔈 Speaking: {text}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                filename = tmpfile.name
                asyncio.run(self.synthesize_to_file(text, filename))

            audio = AudioSegment.from_file(filename)
            os.remove(filename)
            self._play_audio_segment(audio)

        except Exception as e:
            logger.error(f"Speech error: {e}")

    def speak(self, text):
        if not text:
            return

        self.stop_current()  # Interrupt current playback

        # Clear queue and add new text
        with self.lock:
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except queue.Empty:
                    break
            self.tts_queue.put(text)

    def shutdown(self):
        logger.info("Shutting down TTS Player")
        self.stop_current()
        self.stop_event.set()
        # Wait for the queue to be fully processed
        try:
            self.tts_queue.join(timeout=1.0)
        except Exception as e:
            # If we time out, just continue with shutdown
            logger.error(f"Error joining TTS queue: {e}")
