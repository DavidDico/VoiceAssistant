import pyaudio
import pvporcupine
import struct
import os
import queue
import threading
from google.cloud import speech
from openai import OpenAI
import wave
import tempfile
import json
import numpy as np
from assistant_tools import AssistantTools, TOOL_FUNCTIONS

class VoiceAssistant:
    def __init__(self, porcupine_access_key, openai_api_key, google_credentials_path=None, openweather_api_key=None):
        """
        Initialize the voice assistant.
        
        Args:
            porcupine_access_key: Your Picovoice access key
            openai_api_key: Your OpenAI API key
            google_credentials_path: Path to Google Cloud credentials JSON (optional if set in env)
            openweather_api_key: OpenWeatherMap API key (optional, for weather features)
        """
        self.porcupine_access_key = porcupine_access_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize tools
        self.tools = AssistantTools(openweather_api_key=openweather_api_key)
        
        # Set Google credentials if provided
        if google_credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
        
        self.google_speech_client = speech.SpeechClient()
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 512
        
        # State
        self.listening_for_command = False
        self.audio_queue = queue.Queue()
        
        # System prompt for GPT
        self.system_prompt = """Tu es un assistant vocal utile. Réponds toujours en français. 
        Garde tes réponses concises et conversationnelles, car elles seront prononcées à voix haute. 
        Vise des réponses de moins de 3 phrases sauf si plus de détails sont spécifiquement demandés."""
        
        # Conversation history (optional - keeps context)
        self.conversation_history = []
        
    def initialize_porcupine(self):
        """Initialize Porcupine with a wake word."""
        # Using built-in "porcupine" wake word - you can change this
        # Options: 'alexa', 'americano', 'blueberry', 'bumblebee', 'computer', 
        #          'grapefruit', 'grasshopper', 'hey google', 'hey siri', 
        #          'jarvis', 'ok google', 'picovoice', 'porcupine', 'terminator'
        self.porcupine = pvporcupine.create(
            access_key=self.porcupine_access_key,
            keywords=['porcupine']
        )
        print(f"Porcupine initialisé. Longueur de trame: {self.porcupine.frame_length}")
        print(f"Utilisation du mot d'activation: 'porcupine' (parlez ensuite en français)")
        
    def listen_for_wake_word(self):
        """Continuously listen for the wake word using Porcupine."""
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                rate=self.porcupine.sample_rate,
                channels=self.channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            print("\n🎤 Écoute du mot d'activation 'porcupine'...")
            print("(Dites 'porcupine' pour activer)\n")
            
            while True:
                pcm = stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print("✓ Mot d'activation détecté!")
                    self.play_acknowledgment_sound()
                    stream.stop_stream()
                    stream.close()
                    self.listen_for_command()
                    
                    # Restart stream for next wake word
                    stream = audio.open(
                        rate=self.porcupine.sample_rate,
                        channels=self.channels,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.porcupine.frame_length
                    )
                    print("\n🎤 Écoute du mot d'activation 'porcupine'...\n")
                    
        except KeyboardInterrupt:
            print("\nArrêt...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
    def play_acknowledgment_sound(self):
        """Play a simple beep to acknowledge wake word detection."""
        try:
            import pygame
            import numpy as np
            
            # Initialize pygame mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            
            # Generate a pleasant beep sound (440 Hz for 200ms)
            sample_rate = 22050
            duration = 0.2  # seconds
            frequency = 440  # Hz (A note)
            
            # Generate sine wave
            samples = int(sample_rate * duration)
            wave_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
            
            # Apply fade in/out to avoid clicks
            fade_samples = int(sample_rate * 0.02)  # 20ms fade
            wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Convert to 16-bit integers
            wave_data = (wave_data * 32767).astype(np.int16)
            
            # Create stereo sound (duplicate for both channels)
            stereo_data = np.column_stack((wave_data, wave_data))
            
            # Play the sound
            sound = pygame.sndarray.make_sound(stereo_data)
            sound.play()
            
            # Wait for sound to finish
            while pygame.mixer.get_busy():
                pygame.time.wait(10)
            
            pygame.mixer.quit()
            
        except ImportError:
            # Fallback to console beep if pygame/numpy not available
            print("🔔 Beep!")
        except Exception as e:
            # Silent fallback if sound generation fails
            print("🔔")
        
    def listen_for_command(self):
        """Listen for user command using Google Speech-to-Text streaming."""
        print("🎙️  En écoute de votre commande... (parlez maintenant)")
        
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        # Start audio streaming thread
        self.listening_for_command = True
        audio_thread = threading.Thread(target=self._audio_stream_generator, args=(stream,))
        audio_thread.start()
        
        # Configure Google Speech recognition
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="fr-FR",  # French language
            enable_automatic_punctuation=True,
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=True  # Stop after detecting end of speech
        )
        
        # Create streaming request generator
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in self._audio_generator()
        )
        
        try:
            responses = self.google_speech_client.streaming_recognize(
                streaming_config, requests
            )
            
            transcript = self._process_responses(responses)
            
            if transcript:
                print(f"📝 Vous avez dit: {transcript}")
                self.process_command(transcript)
            else:
                print("❌ Aucun discours détecté. Retour à l'écoute du mot d'activation.")
                
        except Exception as e:
            print(f"❌ Erreur lors de la reconnaissance vocale: {e}")
        finally:
            self.listening_for_command = False
            audio_thread.join()
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
    def _audio_stream_generator(self, stream):
        """Generate audio chunks from the microphone."""
        while self.listening_for_command:
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Erreur de lecture audio: {e}")
                break
                
    def _audio_generator(self):
        """Yield audio chunks from the queue for Google Speech API."""
        while self.listening_for_command:
            try:
                chunk = self.audio_queue.get(timeout=1)
                yield chunk
            except queue.Empty:
                continue
                
    def _process_responses(self, responses):
        """Process streaming responses from Google Speech API."""
        transcript = ""
        
        for response in responses:
            if not response.results:
                continue
                
            result = response.results[0]
            
            if not result.alternatives:
                continue
                
            # Show interim results
            if not result.is_final:
                print(f"   (provisoire: {result.alternatives[0].transcript})", end='\r')
            else:
                transcript = result.alternatives[0].transcript
                self.listening_for_command = False
                break
                
        return transcript.strip()
        
    def process_command(self, command):
        """Send command to OpenAI GPT with function calling support and speak the response."""
        print("🤖 Traitement avec GPT...")
        
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": command
            })
            
            # Keep only last 10 messages to avoid token limits
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Call OpenAI API with function calling
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" for better quality
                messages=messages,
                tools=TOOL_FUNCTIONS,
                tool_choice="auto",  # Let the model decide when to use tools
                max_tokens=150,
                temperature=0.7
            )
            
            response_message = response.choices[0].message
            
            # Check if the model wants to call functions
            if response_message.tool_calls:
                # Execute the function calls
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Appel de fonction: {function_name}({function_args})")
                    
                    # Execute the function
                    function_result = self._execute_tool(function_name, function_args)
                    
                    # Add function call and result to conversation
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "content": json.dumps(function_result, ensure_ascii=False),
                        "tool_call_id": tool_call.id
                    })
                
                # Get the final response after function execution
                second_response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": self.system_prompt}] + self.conversation_history,
                    max_tokens=150,
                    temperature=0.7
                )
                
                assistant_message = second_response.choices[0].message.content
            else:
                # No function call needed, use the direct response
                assistant_message = response_message.content
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            print(f"💬 Assistant: {assistant_message}")
            
            # Speak the response
            self.speak(assistant_message)
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de la commande: {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_tool(self, function_name: str, function_args: dict):
        """Execute a tool function and return the result."""
        try:
            if function_name == "get_current_time":
                return self.tools.get_current_time(**function_args)
            elif function_name == "get_weather":
                return self.tools.get_weather(**function_args)
            elif function_name == "calculate":
                return self.tools.calculate(**function_args)
            elif function_name == "set_timer":
                return self.tools.set_timer(**function_args)
            elif function_name == "search_web":
                return self.tools.search_web(**function_args)
            else:
                return {"success": False, "error": f"Fonction inconnue: {function_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def speak(self, text):
        """Convert text to speech using OpenAI TTS and play it."""
        print("🔊 Prononciation de la réponse...")
        
        try:
            # Generate speech using OpenAI TTS
            response = self.openai_client.audio.speech.create(
                model="tts-1",  # or "tts-1-hd" for higher quality
                voice="alloy",  # Good for French: alloy, echo, or nova
                input=text
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                temp_audio.write(response.content)
                temp_audio_path = temp_audio.name
            
            # Play the audio file
            self._play_audio_file(temp_audio_path)
            
            # Clean up - wait a bit to ensure pygame has released the file
            import time
            time.sleep(0.5)
            try:
                os.unlink(temp_audio_path)
            except:
                pass  # Ignore errors if file is still in use
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération/lecture de la parole: {e}")
            
    def _play_audio_file(self, file_path):
        """Play an audio file using PyAudio."""
        try:
            # For MP3, we need to convert or use a different library
            # Simple solution: use pygame for cross-platform audio playback
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Properly cleanup pygame mixer
            pygame.mixer.music.unload()
            pygame.mixer.quit()
                
        except ImportError:
            print("⚠️  pygame n'est pas installé. Installez avec: pip install pygame")
            print(f"Audio sauvegardé dans: {file_path}")
        except Exception as e:
            print(f"Erreur lors de la lecture audio: {e}")
            
    def run(self):
        """Start the voice assistant."""
        print("=" * 60)
        print("🎙️  Assistant Vocal Démarrage...")
        print("=" * 60)
        
        try:
            self.initialize_porcupine()
            self.listen_for_wake_word()
        except KeyboardInterrupt:
            print("\n\nArrêt en cours...")
        finally:
            if hasattr(self, 'porcupine'):
                self.porcupine.delete()
            print("Au revoir! 👋")


if __name__ == "__main__":
    # Import configuration from config.py
    from config import (
        PORCUPINE_ACCESS_KEY,
        OPENAI_API_KEY,
        GOOGLE_CREDENTIALS_PATH,
        OPENWEATHER_API_KEY
    )
    
    # Create and run the assistant
    assistant = VoiceAssistant(
        porcupine_access_key=PORCUPINE_ACCESS_KEY,
        openai_api_key=OPENAI_API_KEY,
        google_credentials_path=GOOGLE_CREDENTIALS_PATH,
        openweather_api_key=OPENWEATHER_API_KEY
    )
    
    assistant.run()
