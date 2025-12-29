import pyaudio
import pvporcupine
import struct
import os
import queue
import threading
import time
from google.cloud import speech
from openai import OpenAI
import wave
import tempfile
import json
import numpy as np
from assistant_tools import AssistantTools, TOOL_FUNCTIONS

# Extended tool functions with conversation control
EXTENDED_TOOL_FUNCTIONS = TOOL_FUNCTIONS + [
    {
        "type": "function",
        "function": {
            "name": "end_conversation_mode",
            "description": "Terminer le mode conversation et retourner au mode d'écoute du mot d'activation. Utiliser cette fonction quand: 1) L'utilisateur dit explicitement qu'il veut terminer la conversation (ex: 'c'est tout', 'merci', 'au revoir', 'fin de conversation'). 2) L'interaction est naturellement terminée après une réponse de l'utilisateur à une question (ex: après qu'il ait dit 'non' à 'voulez-vous autre chose?'). 3) L'utilisateur a clairement fini sa demande et n'a pas besoin de plus d'information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Raison de la fin de conversation (pour debug)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_conversation_mode",
            "description": "Démarrer le mode conversation où l'assistant écoute continuellement sans nécessiter le mot d'activation. Utiliser quand l'utilisateur demande explicitement une conversation (ex: 'parlons', 'discutons', 'j'aimerais discuter', 'mode conversation').",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Raison du démarrage de la conversation (pour debug)"
                    }
                },
                "required": []
            }
        }
    }
]


class VoiceAssistant:
    def __init__(self, porcupine_access_key, openai_api_key, google_credentials_path=None, openweather_api_key=None, tts_voice="alloy", wake_word="porcupine", custom_wake_word_path=None, google_search_api_key=None, google_search_engine_id=None, twilio_account_sid=None, twilio_auth_token=None, twilio_whatsapp_from=None, whatsapp_contacts=None, twilio_sms_from=None, sms_contacts=None, rss_feeds=None, beep_volume=0.3):
        """
        Initialize the voice assistant.
        
        Args:
            porcupine_access_key: Your Picovoice access key
            openai_api_key: Your OpenAI API key
            google_credentials_path: Path to Google Cloud credentials JSON (optional if set in env)
            openweather_api_key: OpenWeatherMap API key (optional, for weather features)
            tts_voice: OpenAI TTS voice (alloy, echo, fable, onyx, nova, shimmer)
            wake_word: Built-in wake word to use
            custom_wake_word_path: Path to custom .ppn file (overrides wake_word if provided)
            google_search_api_key: Google Custom Search API key (optional, for web search)
            google_search_engine_id: Google Custom Search Engine ID (optional, for web search)
            twilio_account_sid: Twilio Account SID (optional, for WhatsApp & SMS)
            twilio_auth_token: Twilio Auth Token (optional, for WhatsApp & SMS)
            twilio_whatsapp_from: Twilio WhatsApp number (optional, for WhatsApp)
            whatsapp_contacts: Dictionary of contact names to WhatsApp numbers (optional)
            twilio_sms_from: Twilio phone number for SMS (optional, for SMS)
            sms_contacts: Dictionary of contact names to phone numbers (optional)
            rss_feeds: Dictionary of category names to RSS URLs (optional, for news)
            beep_volume: Volume level for beeps and notification sounds (0.0 to 1.0, default 0.3)
        """
        self.porcupine_access_key = porcupine_access_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize tools
        self.tools = AssistantTools(
            openweather_api_key=openweather_api_key,
            google_search_api_key=google_search_api_key,
            google_search_engine_id=google_search_engine_id,
            twilio_account_sid=twilio_account_sid,
            twilio_auth_token=twilio_auth_token,
            twilio_whatsapp_from=twilio_whatsapp_from,
            whatsapp_contacts=whatsapp_contacts,
            twilio_sms_from=twilio_sms_from,
            sms_contacts=sms_contacts,
            rss_feeds=rss_feeds
        )
        self.tools.assistant = self  # Give tools access to assistant for timer announcements
        
        # TTS settings
        self.tts_voice = tts_voice
        
        # Audio settings
        self.beep_volume = max(0.0, min(1.0, beep_volume))  # Clamp between 0 and 1
        
        # Wake word settings
        self.wake_word = wake_word
        self.custom_wake_word_path = custom_wake_word_path
        
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
        self._is_speaking = False  # Track if assistant is currently speaking
        self._audio_lock = threading.Lock()  # Lock to prevent simultaneous audio playback
        
        # Conversation mode state
        self.conversation_mode = False
        self._conversation_should_end = False  # Flag set by GPT to end conversation
        
        # System prompt for GPT
        import datetime
        now = datetime.datetime.now()
        current_date = now.strftime("%A %d %B %Y")  # e.g., "Vendredi 27 Décembre 2024"
        current_time = now.strftime("%H:%M")  # e.g., "14:30"
        
        # Get the wake word name for the prompt
        wake_word_name = self.wake_word if not self.custom_wake_word_path else os.path.basename(self.custom_wake_word_path).replace('.ppn', '').replace('_', ' ')
        
        # Get the first contact names for "moi" references
        first_whatsapp_contact = list(whatsapp_contacts.keys())[0] if whatsapp_contacts else None
        first_sms_contact = list(sms_contacts.keys())[0] if sms_contacts else None
        
        self.system_prompt = f"""Tu es un assistant vocal utile appelé "{wake_word_name}". Réponds toujours en français. 
        Garde tes réponses concises et conversationnelles, car elles seront prononcées à voix haute. 
        Vise des réponses de moins de 3 phrases sauf si plus de détails sont spécifiquement demandés.
        
        INFORMATION IMPORTANTE: Nous sommes le {current_date} et il est {current_time}.
        Utilise cette information pour répondre aux questions sur "aujourd'hui", "hier", "demain", etc.
        
        MESSAGES SMS ET WHATSAPP:
        Quand l'utilisateur dit "envoie-moi un SMS", "envoie-moi un message", "envoie-moi un WhatsApp" ou utilise 
        "moi/me" comme destinataire:
        - Pour WhatsApp: utilise "{first_whatsapp_contact}" comme nom de contact
        - Pour SMS: utilise "{first_sms_contact}" comme nom de contact
        
        MODE CONVERSATION:
        Tu as accès à deux fonctions spéciales pour gérer le mode conversation.
        
        1. start_conversation_mode: UTILISE CETTE FONCTION QUAND:
           - L'utilisateur demande explicitement une conversation (ex: "parlons", "discutons", "mode conversation")
           - Tu poses une question qui ATTEND UNE RÉPONSE de l'utilisateur (ex: "Veux-tu ajouter autre chose?", "Quel format préfères-tu?")
           - Tu as besoin d'une clarification pour accomplir une tâche
           
           IMPORTANT: Si tu termines ta réponse par une question, tu DOIS appeler start_conversation_mode pour que l'utilisateur puisse répondre!
        
        2. end_conversation_mode: UTILISE CETTE FONCTION QUAND:
           - L'utilisateur dit vouloir terminer (ex: "c'est tout", "merci", "non", "au revoir", "fin", "stop")
           - L'utilisateur répond négativement à ta question (ex: "non", "non merci", "c'est bon")
           - L'interaction est naturellement terminée et tu n'as plus de question à poser
        
        RÈGLE IMPORTANTE: Si tu poses une question → appelle start_conversation_mode
                          Si l'utilisateur répond et tu n'as plus de question → appelle end_conversation_mode
        
        Si l'utilisateur mentionne ton nom ("{wake_word_name}") pendant la conversation, c'est normal - 
        il peut t'appeler par ton nom même en mode conversation."""
        
        # Conversation history (optional - keeps context)
        self.conversation_history = []
        
        # Speech recognition settings
        self.silence_timeout = 1.5  # Seconds of silence before considering speech ended
        self.last_transcript_time = None
        
    def initialize_porcupine(self):
        """Initialize Porcupine with a wake word."""
        if self.custom_wake_word_path:
            # Use custom wake word file
            # For French wake words, we need to specify the French model
            import os
            import pvporcupine
            
            # Get the Porcupine package directory
            porcupine_dir = os.path.dirname(pvporcupine.__file__)
            
            # Path to French model (included in pvporcupine package)
            french_model_path = os.path.join(porcupine_dir, "lib", "common", "porcupine_params_fr.pv")
            
            # Check if French model exists, otherwise use default
            if os.path.exists(french_model_path):
                print(f"Utilisation du modèle français: {french_model_path}")
                self.porcupine = pvporcupine.create(
                    access_key=self.porcupine_access_key,
                    keyword_paths=[self.custom_wake_word_path],
                    model_path=french_model_path
                )
            else:
                print(f"⚠️ Modèle français non trouvé à: {french_model_path}")
                print(f"Tentative sans spécifier le modèle...")
                # Try without model_path and let Porcupine auto-detect
                self.porcupine = pvporcupine.create(
                    access_key=self.porcupine_access_key,
                    keyword_paths=[self.custom_wake_word_path]
                )
            
            print(f"Porcupine initialisé. Longueur de trame: {self.porcupine.frame_length}")
            print(f"Utilisation du mot d'activation personnalisé (parlez ensuite en français)")
        else:
            # Use built-in wake word
            self.porcupine = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=[self.wake_word]
            )
            print(f"Porcupine initialisé. Longueur de trame: {self.porcupine.frame_length}")
            print(f"Utilisation du mot d'activation: '{self.wake_word}' (parlez ensuite en français)")
        
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
            
            print(f"\n🎤 Écoute du mot d'activation '{self.wake_word}'...")
            print(f"(Dites '{self.wake_word}' pour activer)\n")
            
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
                    
                    # Check if we entered conversation mode
                    if self.conversation_mode:
                        self._run_conversation_loop()
                    
                    # Restart stream for next wake word
                    stream = audio.open(
                        rate=self.porcupine.sample_rate,
                        channels=self.channels,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.porcupine.frame_length
                    )
                    # Play wake word mode sound to indicate we're back to requiring wake word
                    self.play_wake_word_mode_sound()
                    print(f"\n🎤 Écoute du mot d'activation '{self.wake_word}'...\n")
                    
        except KeyboardInterrupt:
            print("\nArrêt...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
    
    def _run_conversation_loop(self):
        """Run the conversation mode loop until it ends."""
        print("\n💬 Mode conversation activé - Je vous écoute sans mot d'activation")
        print("   (Dites 'stop', 'fin', ou 'au revoir' pour terminer)\n")
        
        while self.conversation_mode:
            # Check if GPT signaled end of conversation
            if self._conversation_should_end:
                print("\n✓ Conversation terminée par l'assistant")
                self.conversation_mode = False
                self._conversation_should_end = False
                break
            
            # Play a soft beep to indicate we're listening
            self.play_listening_beep()
            
            # Listen for next command
            print("🎙️ Je vous écoute...")
            self.listen_for_command()
        
        print("👋 Retour au mode mot d'activation\n")
            
    def play_acknowledgment_sound(self):
        """Play a simple beep to acknowledge wake word detection."""
        # Acquire lock to prevent simultaneous audio playback
        with self._audio_lock:
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
                
                # Apply volume
                wave_data *= self.beep_volume
                
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
    
    def play_wake_word_mode_sound(self):
        """Play a descending tone to indicate return to wake word mode."""
        with self._audio_lock:
            try:
                import pygame
                import numpy as np
                
                # Initialize pygame mixer
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                
                sample_rate = 22050
                
                # Generate two descending notes (A5 -> E5)
                frequencies = [880, 659]  # A5 to E5
                duration_per_note = 0.15  # seconds
                
                all_notes = []
                for freq in frequencies:
                    samples = int(sample_rate * duration_per_note)
                    wave_data = np.sin(2 * np.pi * freq * np.linspace(0, duration_per_note, samples))
                    
                    # Apply fade in/out
                    fade_samples = int(sample_rate * 0.01)  # 10ms fade
                    wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                    
                    all_notes.append(wave_data)
                
                # Combine notes
                combined = np.concatenate(all_notes)
                
                # Apply volume
                combined *= self.beep_volume
                
                # Convert to 16-bit integers
                combined = (combined * 32767).astype(np.int16)
                
                # Create stereo sound
                stereo_data = np.column_stack((combined, combined))
                
                # Play the sound
                sound = pygame.sndarray.make_sound(stereo_data)
                sound.play()
                
                # Wait for sound to finish
                while pygame.mixer.get_busy():
                    pygame.time.wait(10)
                
                pygame.mixer.quit()
                
            except ImportError:
                print("🔔⬇️")
            except Exception as e:
                print("🔔⬇️")
    
    def play_conversation_mode_sound(self):
        """Play an ascending tone to indicate conversation mode is active."""
        with self._audio_lock:
            try:
                import pygame
                import numpy as np
                
                # Initialize pygame mixer
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                
                sample_rate = 22050
                
                # Generate two ascending notes (E5 -> A5)
                frequencies = [659, 880]  # E5 to A5
                duration_per_note = 0.15  # seconds
                
                all_notes = []
                for freq in frequencies:
                    samples = int(sample_rate * duration_per_note)
                    wave_data = np.sin(2 * np.pi * freq * np.linspace(0, duration_per_note, samples))
                    
                    # Apply fade in/out
                    fade_samples = int(sample_rate * 0.01)  # 10ms fade
                    wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                    
                    all_notes.append(wave_data)
                
                # Combine notes
                combined = np.concatenate(all_notes)
                
                # Apply volume
                combined *= self.beep_volume
                
                # Convert to 16-bit integers
                combined = (combined * 32767).astype(np.int16)
                
                # Create stereo sound
                stereo_data = np.column_stack((combined, combined))
                
                # Play the sound
                sound = pygame.sndarray.make_sound(stereo_data)
                sound.play()
                
                # Wait for sound to finish
                while pygame.mixer.get_busy():
                    pygame.time.wait(10)
                
                pygame.mixer.quit()
                
            except ImportError:
                print("🔔⬆️")
            except Exception as e:
                print("🔔⬆️")
    
    def play_listening_beep(self):
        """Play a soft, short beep to indicate the assistant is listening for a response."""
        with self._audio_lock:
            try:
                import pygame
                import numpy as np
                
                # Initialize pygame mixer
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                
                sample_rate = 22050
                duration = 0.1  # Very short
                frequency = 523  # C5 - a soft, neutral note
                
                samples = int(sample_rate * duration)
                wave_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
                
                # Apply fade in/out
                fade_samples = int(sample_rate * 0.02)
                wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
                wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                # Apply volume (listening beep is slightly quieter than others)
                wave_data *= self.beep_volume * 0.7
                
                # Convert to 16-bit integers
                wave_data = (wave_data * 32767).astype(np.int16)
                
                # Create stereo sound
                stereo_data = np.column_stack((wave_data, wave_data))
                
                # Play the sound
                sound = pygame.sndarray.make_sound(stereo_data)
                sound.play()
                
                # Wait for sound to finish
                while pygame.mixer.get_busy():
                    pygame.time.wait(10)
                
                pygame.mixer.quit()
                
            except Exception:
                pass  # Silent fail
        
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
            # single_utterance removed - will listen longer
            # You can add a manual timeout instead
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
                # Check for sentinel value indicating we should stop
                if chunk is None:
                    return
                yield chunk
            except queue.Empty:
                continue
                
    def _process_responses(self, responses):
        """Process streaming responses from Google Speech API."""
        import time
        import threading
        
        transcript = ""
        last_interim_transcript = ""
        last_final_transcript = ""
        last_activity_time = [time.time()]  # Use list to allow modification in nested function
        should_stop = [False]
        got_any_speech = [False]  # Track if we've received any speech at all
        
        def timeout_checker():
            """Check for timeout in a separate thread."""
            while not should_stop[0]:
                # Stop if we got speech and then silence for too long
                if got_any_speech[0] and time.time() - last_activity_time[0] > self.silence_timeout:
                    print(f"\n   [Silence détectée - fin de commande]")
                    self.listening_for_command = False
                    should_stop[0] = True
                    break
                time.sleep(0.1)  # Check every 100ms
        
        # Start timeout checker thread
        timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
        timeout_thread.start()
        
        try:
            for response in responses:
                if should_stop[0]:
                    break
                    
                if not response.results:
                    continue
                    
                result = response.results[0]
                
                if not result.alternatives:
                    continue
                
                # Mark that we got some speech
                got_any_speech[0] = True
                    
                # Update activity time whenever we get any result
                last_activity_time[0] = time.time()
                    
                # Show interim results
                if not result.is_final:
                    last_interim_transcript = result.alternatives[0].transcript
                    print(f"   (provisoire: {last_interim_transcript})", end='\r')
                else:
                    # Got a final result
                    last_final_transcript = result.alternatives[0].transcript
                    transcript = last_final_transcript
                    print(f"\n   [Confirmé: {transcript}]")
                    
        finally:
            should_stop[0] = True
            timeout_thread.join(timeout=0.5)
        
        # If we have a final transcript, use it. Otherwise use the last interim result
        if transcript:
            return transcript.strip()
        elif last_interim_transcript:
            print(f"\n   [Utilisation du résultat provisoire: {last_interim_transcript}]")
            return last_interim_transcript.strip()
        else:
            return ""
        
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
            
            # Loop until GPT gives a final response without tool calls
            max_iterations = 5  # Safety limit to prevent infinite loops
            iteration = 0
            assistant_message = None
            
            while iteration < max_iterations:
                iteration += 1
                
                # Call OpenAI API with function calling (including conversation control functions)
                messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=EXTENDED_TOOL_FUNCTIONS,
                    tool_choice="auto",
                    max_tokens=150,
                    temperature=0.7
                )
                
                response_message = response.choices[0].message
                
                # If no tool calls, we have our final response
                if not response_message.tool_calls:
                    assistant_message = response_message.content
                    break
                
                # Process all tool calls from this response
                # First, add the assistant message with all tool calls
                tool_calls_for_history = []
                for tool_call in response_message.tool_calls:
                    tool_calls_for_history.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_for_history
                })
                
                # Execute each tool and add results
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Appel de fonction: {function_name}({function_args})")
                    
                    # Handle conversation control functions
                    if function_name == "end_conversation_mode":
                        print(f"   → Fin du mode conversation: {function_args.get('reason', 'non spécifié')}")
                        self._conversation_should_end = True
                        function_result = {"success": True, "message": "Mode conversation terminé"}
                    elif function_name == "start_conversation_mode":
                        print(f"   → Démarrage mode conversation: {function_args.get('reason', 'non spécifié')}")
                        self.conversation_mode = True
                        self._conversation_should_end = False
                        self.play_conversation_mode_sound()
                        function_result = {"success": True, "message": "Mode conversation activé"}
                    else:
                        # Execute other tool functions
                        function_result = self._execute_tool(function_name, function_args)
                    
                    # Add tool result to conversation
                    self.conversation_history.append({
                        "role": "tool",
                        "content": json.dumps(function_result, ensure_ascii=False),
                        "tool_call_id": tool_call.id
                    })
                
                # Continue loop to get next response (which may have more tool calls or final message)
            
            # Add assistant response to history
            if assistant_message:
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
            elif function_name == "send_whatsapp_message":
                return self.tools.send_whatsapp_message(**function_args)
            elif function_name == "send_sms":
                return self.tools.send_sms(**function_args)
            elif function_name == "get_news":
                return self.tools.get_news(**function_args)
            else:
                return {"success": False, "error": f"Fonction inconnue: {function_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def speak(self, text):
        """Convert text to speech using OpenAI TTS and play it."""
        print("🔊 Prononciation de la réponse...")
        
        # Acquire lock to prevent simultaneous audio playback
        with self._audio_lock:
            self._is_speaking = True  # Mark that we're speaking
            
            try:
                # Wake up wireless headset with silent audio
                self._play_headset_wakeup()
                
                # Generate speech using OpenAI TTS
                response = self.openai_client.audio.speech.create(
                    model="tts-1",  # or "tts-1-hd" for higher quality
                    voice=self.tts_voice,
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
            finally:
                self._is_speaking = False  # Mark that we're done speaking
            
    def _play_headset_wakeup(self):
        """Play a very brief, nearly silent tone to wake up wireless headsets."""
        try:
            import pygame
            
            # Initialize mixer if not already done
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Generate a very short, very quiet tone (50ms at low volume)
            sample_rate = 22050
            duration = 0.05  # 50 milliseconds
            frequency = 440
            
            samples = int(sample_rate * duration)
            wave_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
            
            # Make it very quiet (1% volume)
            wave_data = wave_data * 0.01
            
            # Convert to 16-bit integers
            wave_data = (wave_data * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo_data = np.column_stack((wave_data, wave_data))
            
            # Play the sound
            sound = pygame.sndarray.make_sound(stereo_data)
            sound.play()
            
            # Wait for it to finish
            while pygame.mixer.get_busy():
                pygame.time.wait(10)
            
            pygame.mixer.quit()
            
        except Exception:
            # Silently fail - not critical
            pass
    
    def _play_audio_file(self, file_path):
        """Play an audio file using pygame."""
        try:
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
        print("\n📋 Fonctionnalités du mode conversation:")
        print("   • Dites 'parlons' ou 'mode conversation' pour activer")
        print("   • Dites 'stop', 'fin', ou 'au revoir' pour terminer")
        print("   • L'assistant terminera la conversation quand l'échange est fini\n")
        
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
        OPENWEATHER_API_KEY,
        TTS_VOICE,
        WAKE_WORD,
        CUSTOM_WAKE_WORD_PATH,
        GOOGLE_SEARCH_API_KEY,
        GOOGLE_SEARCH_ENGINE_ID,
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
        TWILIO_WHATSAPP_FROM,
        WHATSAPP_CONTACTS,
        TWILIO_SMS_FROM,
        SMS_CONTACTS,
        RSS_FEEDS
    )
    
    # Optional: import BEEP_VOLUME if defined in config, otherwise use default
    try:
        from config import BEEP_VOLUME
    except ImportError:
        BEEP_VOLUME = 0.3  # Default value
    
    # Create and run the assistant
    assistant = VoiceAssistant(
        porcupine_access_key=PORCUPINE_ACCESS_KEY,
        openai_api_key=OPENAI_API_KEY,
        google_credentials_path=GOOGLE_CREDENTIALS_PATH,
        openweather_api_key=OPENWEATHER_API_KEY,
        tts_voice=TTS_VOICE,
        wake_word=WAKE_WORD,
        custom_wake_word_path=CUSTOM_WAKE_WORD_PATH,
        google_search_api_key=GOOGLE_SEARCH_API_KEY,
        google_search_engine_id=GOOGLE_SEARCH_ENGINE_ID,
        twilio_account_sid=TWILIO_ACCOUNT_SID,
        twilio_auth_token=TWILIO_AUTH_TOKEN,
        twilio_whatsapp_from=TWILIO_WHATSAPP_FROM,
        whatsapp_contacts=WHATSAPP_CONTACTS,
        twilio_sms_from=TWILIO_SMS_FROM,
        sms_contacts=SMS_CONTACTS,
        rss_feeds=RSS_FEEDS,
        beep_volume=BEEP_VOLUME
    )
    
    assistant.run()
