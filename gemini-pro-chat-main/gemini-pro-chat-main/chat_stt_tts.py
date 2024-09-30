import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
import re
import asyncio
from pydub import AudioSegment  # For audio playback
from pydub.playback import _play_with_simpleaudio  # For non-blocking playback
import speech_recognition as sr  # For Speech-to-Text
import threading
import edge_tts


class ChatHistoryManager:
    def __init__(self, filename="chat_history.txt", max_file_size_mb=5):
        self.history = []
        self.filename = filename
        self.max_file_size_mb = max_file_size_mb
        self.interrupt_flag = threading.Event()  # Event to interrupt playback
        self.playback_obj = None  # Reference to the playback object

    def add_message(self, role, text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append(
            {'role': role, 'text': text, 'timestamp': timestamp})

    def save_to_file(self):
        self._rotate_file_if_needed()
        with open(self.filename, "a", encoding="utf-8") as file:
            for message in self.history:
                file.write(
                    f"{message['timestamp']} {message['role']}: {message['text']}\n")
        self.history.clear()

    def display(self):
        for message in self.history:
            print(
                f"{message['timestamp']} {message['role']}: {message['text']}")

    def _rotate_file_if_needed(self):
        if not os.path.exists(self.filename):
            with open(self.filename, "a", encoding="utf-8") as file:
                pass

        if os.path.getsize(self.filename) > self.max_file_size_mb * 1024 * 1024:
            os.rename(self.filename, self.filename + ".backup")

    # TTS integration: Convert text to speech and play it
    async def text_to_speech(self, text, output_file="response.mp3", voice="en-GB-LibbyNeural"):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            print(f"TTS: Saved response as {output_file}")

            # Play the MP3 file asynchronously and allow interruption
            playback_thread = threading.Thread(target=self.play_audio, args=(output_file,))
            playback_thread.start()

            # Listen for the "stop" command in parallel
            await self.listen_for_stop_command()
            playback_thread.join()

        except Exception as e:
            print(f"Failed to save or play TTS: {e}")

    def play_audio(self, file):
        """Plays audio and stops if interrupt_flag is set"""
        try:
            self.interrupt_flag.clear()  # Reset the interrupt flag

            # Load the audio using pydub
            audio = AudioSegment.from_file(file, format="mp3")

            # Start playback using simpleaudio
            self.playback_obj = _play_with_simpleaudio(audio)

            # Wait for the playback to finish or be interrupted
            while self.playback_obj.is_playing():
                if self.interrupt_flag.is_set():
                    print("Stopping audio playback...")
                    self.playback_obj.stop()  # Stop playback
                    break
        except Exception as e:
            print(f"Failed to play sound: {e}")

    async def listen_for_stop_command(self):
        """Listen for 'stop' command while playing audio"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening for 'stop' command...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                # Listen in the background and wait for "stop" to interrupt playback
                while not self.interrupt_flag.is_set():
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    text = recognizer.recognize_google(audio).lower()
                    if "stop" in text:
                        print("Stop command detected.")
                        self.interrupt_flag.set()  # Signal to stop playback
                        break
            except sr.UnknownValueError:
                print("Didn't catch that, no 'stop' detected.")
            except sr.RequestError:
                print("Speech recognition service failed.")
            except sr.WaitTimeoutError:
                print("Timeout, no speech detected.")

    # STT integration: Convert speech to text
    def speech_to_text(self, timeout=5, phrase_time_limit=5):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening for speech...")
            recognizer.adjust_for_ambient_noise(source)
            
            try:
                # Listen for the user's input with a timeout and phrase time limit
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
                # Recognize the speech and convert it to text
                print("Recognizing speech...")
                text = recognizer.recognize_google(audio)
                print(f"User (speech): {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I did not understand the speech.")
                return None
            except sr.RequestError:
                print("Could not request results from the speech recognition service.")
                return None
            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase.")
                return None


async def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please set your GEMINI_API_KEY in the environment.")

    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

    history_manager = ChatHistoryManager()
    history_manager.add_message("system", "--- New Session ---")

    model = genai.GenerativeModel(
        'gemini-pro', generation_config=generation_config, safety_settings=safety_settings)
    chat = model.start_chat(history=[])

    while True:
        try:
            # Convert speech to text (STT) using the microphone
            history_manager.add_message("system", "Listening for speech input...")
            user_input = history_manager.speech_to_text()

            # If no valid input, continue the loop
            if not user_input:
                print("Could not capture speech or no input given.")
                continue

            # Add the user's message to the chat history
            history_manager.add_message("user", user_input)

            # Process the input and get a response from the model
            response = chat.send_message(user_input, stream=True)
            response_text = ""
            for chunk in response:
                if chunk.text.endswith("."):
                    response_text += chunk.text
                else:
                    response_text += re.sub(r'\s*$', '.', chunk.text)
                print(chunk.text)

            # Add the model's response to the chat history
            history_manager.add_message("gemini", response_text)

            # Convert the AI's response text to speech and save it
            await history_manager.text_to_speech(response_text, output_file="response.mp3")

        except KeyboardInterrupt:
            print("\nUser interrupted the script. Exiting...")
            break

        except Exception as e:
            print(f"An error occurred: {e}")

# Run the main event loop
if __name__ == "__main__":
    asyncio.run(main())
