"""
DesktopAI Assistant - Clean and Modular Voice Assistant

This is the main entry point for the DesktopAI assistant with a clean modular architecture.
"""

import os
import sys

# Add the src directory to Python path for easy imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import threading

from src.audio.audio_processor import AudioProcessor
from src.audio.listener import Listener
from src.audio.ttsplayer import TTSPlayer
from src.core.assistant import call_agent
from src.core.tools import register_stop_assistant
from src.ui.overlay import app, overlay
from src.utils.logger import get_logger
from src.utils.thread_executor import executor

logger = get_logger()


class DesktopAssistant:
    """Main DesktopAI Assistant class with clean modular architecture."""

    def __init__(self):
        """Initialize the assistant with all required components."""
        self.lock = threading.Lock()
        self._setup_components()
        self._setup_ui()

    def _setup_components(self):
        """Initialize core components."""
        # Text-to-Speech
        self.speech = TTSPlayer()
        self.speech.start()

        # Audio processing
        self.audio_processor = AudioProcessor()
        self.listener = Listener(tts_player=self.speech, overlay=overlay)

    def _setup_ui(self):
        """Setup user interface."""
        overlay.on_new_message = self.process_query
        overlay.start()

    def process_audio(self, audio):
        """Process incoming audio data."""
        with self.lock:
            logger.info("Processing audio...")
            overlay.put_message("status", "Analyzing voice...", "gold")
            transcription = self.audio_processor.process_audio(audio)
            self.process_query(transcription)

    def process_query(self, query: str):
        """Process the text query."""
        overlay.put_message("query", query)
        logger.debug(f"Invoking agent with: {query}")
        overlay.put_message("status", "Processing...", "gold")
        try:
            response = call_agent(query)
            logger.info(f"Agent response: {response}")
            overlay.put_message("response", response)
            overlay.put_message("status", "Active", "green")
            self.speech.speak(response)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            overlay.put_message("status", "Error occurred", "red")
            self.speech.speak("Sorry, an error occurred while processing your request.")

    def start(self):
        """Start the assistant."""
        logger.info("🔊 Listening in the background... Say something!")
        overlay.put_message("status", "Active", "green")
        executor.submit(self.listener.listen, self.process_audio)

    def shutdown(self):
        """Clean shutdown of the assistant."""
        self.speech.speak("Shutting down!")
        overlay.put_message("status", "Shutting down...", "red")
        logger.info("Shutting down...")

        self.listener.stop_listening()
        self.speech.shutdown()
        executor.shutdown(wait=False, cancel_futures=True)
        overlay.close()


def main():
    """Main function to run Desktop Assistant."""
    assistant = DesktopAssistant()
    register_stop_assistant(assistant.shutdown)
    assistant.start()
    app.exec()


if __name__ == "__main__":
    main()
