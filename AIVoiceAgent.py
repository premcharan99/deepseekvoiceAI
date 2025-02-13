from elevenlabs.client import ElevenLabs
from elevenlabs import save
import ollama
import pyaudio
import os
import json
from vosk import Model, KaldiRecognizer
import wget  

class AIVoiceAgent:
    def __init__(self):
        # Set ElevenLabs API Key
        self.client = ElevenLabs(
            api_key="sk_23162e1c986e0462ed494bf11483e0d3cebd96a1d279600f"
        )

        # Initialize Vosk model
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print("Downloading Vosk model...")
            wget.download("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
            # Unzip the model
            import zipfile
            with zipfile.ZipFile("vosk-model-small-en-us-0.15.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("vosk-model-small-en-us-0.15.zip")
            
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        
        self.full_transcript = [
            {"role": "system", "content": "You are a helpful AI assistant. Keep responses under 300 characters."},
        ]

    def start_transcription(self):
        print("\nSpeak now (press Ctrl+C to stop)...")
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096
        )

        try:
            while True:
                data = self.stream.read(4096)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if 'text' in result and len(result['text']) > 0:
                        print(f"\nUser: {result['text']}")
                        self.generate_ai_response(result['text'])
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

    def generate_ai_response(self, text):
        self.full_transcript.append({"role": "user", "content": text})

        ollama_stream = ollama.chat(
            model="deepseek-r1:7b",
            messages=self.full_transcript,
            stream=True,
        )

        print("AI:", end=" ")
        full_response = ""
        for chunk in ollama_stream:
            text_chunk = chunk['message']['content']
            print(text_chunk, end="", flush=True)
            full_response += text_chunk

        audio_stream = self.client.generate(
            text=full_response,
            model="eleven_turbo_v2",
            stream=True
        )
        save(audio_stream, "response.mp3")
        os.system("start response.mp3" if os.name == "nt" else "open response.mp3")

        self.full_transcript.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    # Install required packages if not already installed
    required_packages = ["vosk", "pyaudio", "wget", "elevenlabs", "ollama"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    ai_voice_agent = AIVoiceAgent()
    ai_voice_agent.start_transcription()