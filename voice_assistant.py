import pyaudio
import pvporcupine
import struct
import os
import queue
import threading
import time
import random
from google.cloud import speech
from openai import OpenAI
import wave
import tempfile
import json
import numpy as np
from assistant_tools import AssistantTools, TOOL_FUNCTIONS

# =============================================================================
# EDITABLE GREETINGS - Modify this list to change wake word responses
# =============================================================================
WAKE_WORD_GREETINGS = [
    "Oui ?",
    "Je t'écoute",
    "À ton service",
    "Que puis-je faire pour toi ?",
    "Je suis là",
    "Dis-moi",
    "Comment puis-je t'aider ?",
]
# =============================================================================

# =============================================================================
# FILLER PHRASES - Played while waiting for function calls
# =============================================================================
FILLER_PHRASES = [
    "Hmm...",
    "Ok...",
    "Alors...",
    "Voyons voir...",
    "Euh...",
    "Une seconde...",
    "Je regarde...",
]
# =============================================================================

# Extended tool functions with conversation control
EXTENDED_TOOL_FUNCTIONS = TOOL_FUNCTIONS + [
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": "Terminer la conversation. Appeler cette fonction quand l'utilisateur a fini: après 'merci', 'ok', 'parfait', 'super', 'd'accord', 'au revoir', 'c'est tout', 'non merci', ou tout autre signal de fin.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enable_boost_mode",
            "description": "Activer le mode intelligence augmentée (GPT-4o). Utiliser quand l'utilisateur demande d'être plus intelligent, plus performant, de réfléchir mieux, d'utiliser un meilleur modèle, ou dit des choses comme 'sois plus malin', 'réfléchis mieux', 'mode turbo', 'boost', etc.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "disable_boost_mode",
            "description": "Désactiver le mode intelligence augmentée et revenir au mode normal. Utiliser quand l'utilisateur demande de revenir en mode normal, d'arrêter le boost, de désactiver le turbo, ou dit des choses comme 'mode normal', 'arrête le boost', 'reviens en mode standard', etc.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Functions that should NOT trigger filler phrases (fast/instant functions)
NO_FILLER_FUNCTIONS = {
    "end_conversation",
    "enable_boost_mode",
    "disable_boost_mode",
}


class VoiceAssistant:
    def __init__(self, porcupine_access_key, openai_api_key, google_credentials_path=None, openweather_api_key=None, tts_voice="alloy", wake_word="porcupine", custom_wake_word_path=None, porcupine_model_path=None, google_search_api_key=None, google_search_engine_id=None, twilio_account_sid=None, twilio_auth_token=None, twilio_sms_from=None, sms_contacts=None, rss_feeds=None, beep_volume=0.3, google_calendar_credentials_path=None, google_calendar_id=None, greetings_cache_dir=None, max_history_items=50, history_max_age_seconds=3600, memory_file_path=None, default_city=None, email_smtp_server=None, email_smtp_port=None, email_address=None, email_password=None, email_contacts=None, silence_timeout=1.5, conversation_timeout=10.0):
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
            porcupine_model_path: Path to Porcupine language model .pv file (optional, for non-English wake words)
            google_search_api_key: Google Custom Search API key (optional, for web search)
            google_search_engine_id: Google Custom Search Engine ID (optional, for web search)
            twilio_account_sid: Twilio Account SID (optional, for SMS)
            twilio_auth_token: Twilio Auth Token (optional, for SMS)
            twilio_sms_from: Twilio phone number for SMS (optional, for SMS)
            sms_contacts: Dictionary of contact names to phone numbers (optional)
            rss_feeds: Dictionary of category names to RSS URLs (optional, for news)
            beep_volume: Volume level for beeps and notification sounds (0.0 to 1.0, default 0.3)
            google_calendar_credentials_path: Path to Google Calendar OAuth credentials JSON (optional)
            google_calendar_id: Google Calendar ID to use (optional, defaults to 'primary')
            greetings_cache_dir: Directory to cache greeting audio files (optional, defaults to ~/.casimir_greetings)
            max_history_items: Maximum number of history items to keep (default 50)
            history_max_age_seconds: Maximum age of history items in seconds (default 3600 = 1 hour)
            memory_file_path: Path to memory file (optional, defaults to ~/.casimir_memory.txt)
            default_city: Default city for weather queries (optional, e.g., "Paris")
            email_smtp_server: SMTP server address (optional, e.g., 'smtp.gmail.com')
            email_smtp_port: SMTP server port (optional, e.g., 587)
            email_address: Email address to send from (optional)
            email_password: Email password or app password (optional)
            email_contacts: Dictionary of contact names to email addresses (optional)
            silence_timeout: Seconds of silence before considering speech ended (default 1.5)
            conversation_timeout: Seconds of no speech before ending conversation (default 10.0)
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
            twilio_sms_from=twilio_sms_from,
            sms_contacts=sms_contacts,
            rss_feeds=rss_feeds,
            google_calendar_credentials_path=google_calendar_credentials_path,
            google_calendar_id=google_calendar_id,
            memory_file_path=memory_file_path,
            email_smtp_server=email_smtp_server,
            email_smtp_port=email_smtp_port,
            email_address=email_address,
            email_password=email_password,
            email_contacts=email_contacts
        )
        self.tools.assistant = self  # Give tools access to assistant for timer announcements
        
        # TTS settings
        self.tts_voice = tts_voice
        
        # Audio settings
        self.beep_volume = max(0.0, min(1.0, beep_volume))  # Clamp between 0 and 1
        
        # Wake word settings
        self.wake_word = wake_word
        self.custom_wake_word_path = custom_wake_word_path
        self.porcupine_model_path = porcupine_model_path
        
        # Greetings cache settings
        self.greetings_cache_dir = greetings_cache_dir or os.path.expanduser("~/.casimir_greetings")
        self._greeting_files = {}  # Cache of greeting text -> file path
        self._filler_files = {}    # Cache of filler phrase -> file path
        self._ensure_greetings_cached()
        self._ensure_fillers_cached()
        
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
        
        # Conversation state
        self._conversation_active = False
        self._conversation_should_end = False
        self._boost_mode = False  # When True, use gpt-4o instead of gpt-4o-mini
        
        # Location settings
        self.default_city = default_city
        
        # History configuration
        self.max_history_items = max_history_items
        self.history_max_age_seconds = history_max_age_seconds
        
        # Conversation history with timestamps: list of {"message": {...}, "timestamp": float}
        self.conversation_history = []
        
        # System prompt for GPT
        import datetime
        now = datetime.datetime.now()
        current_date = now.strftime("%A %d %B %Y")  # e.g., "Vendredi 27 Décembre 2024"
        current_time = now.strftime("%H:%M")  # e.g., "14:30"
        
        # Get the wake word name for the prompt
        wake_word_name = self.wake_word if not self.custom_wake_word_path else os.path.basename(self.custom_wake_word_path).replace('.ppn', '').replace('_', ' ')
        
        # Get the first contact names for "moi" references
        first_email_contact = list(email_contacts.keys())[0] if email_contacts else None
        first_sms_contact = list(sms_contacts.keys())[0] if sms_contacts else None
        
        self.system_prompt = f"""Tu es un assistant vocal utile appelé "{wake_word_name}". Réponds toujours en français. 
        Garde tes réponses concises et conversationnelles, car elles seront prononcées à voix haute. 
        Vise des réponses de moins de 3 phrases sauf si plus de détails sont spécifiquement demandés.
        
        IMPORTANT: Ne mentionne JAMAIS d'URLs ou de liens dans tes réponses. Tout ce que tu dis sera lu à voix haute,
        donc les URLs sont inutiles et gênantes. Résume simplement les informations sans proposer de liens.
        
        INFORMATION IMPORTANTE: Nous sommes le {current_date} et il est {current_time}.
        Utilise cette information pour répondre aux questions sur "aujourd'hui", "hier", "demain", etc.
        
        LOCALISATION: {f"L'utilisateur se trouve à {self.default_city}. Utilise {self.default_city} UNIQUEMENT si l'utilisateur demande la météo sans mentionner aucune ville (ex: 'quel temps fait-il ?'). Si l'utilisateur mentionne une ville spécifique (ex: 'météo à Paris', 'temps à Lyon'), utilise TOUJOURS la ville qu'il a mentionnée, pas {self.default_city}." if self.default_city else "Aucune ville par défaut configurée. Demande la ville si l'utilisateur ne la précise pas."}
        
        ACTUALITÉS: Quand tu reçois des actualités, sélectionne les 2-3 plus importantes ou intéressantes et résume-les 
        de façon concise (environ 3 phrases au total). Ne liste pas tous les articles.
        
        MESSAGES EMAIL ET SMS:
        Quand l'utilisateur dit "envoie-moi un email", "envoie-moi un mail", "envoie-moi un message", "envoie-moi un SMS" ou utilise 
        "moi/me" comme destinataire:
        1. Si l'utilisateur s'est identifié dans la conversation (ex: "Je suis David", "C'est Marie", "Moi c'est Pierre"),
           utilise ce nom comme contact pour l'envoi.
        2. Sinon, par défaut:
           - Pour Email: utilise "{first_email_contact}" comme nom de contact
           - Pour SMS: utilise "{first_sms_contact}" comme nom de contact
        
        GESTION DE LA CONVERSATION:
        Tu as une fonction end_conversation pour terminer la conversation.
        
        APPELLE end_conversation quand l'utilisateur dit:
        - "merci", "ok merci", "merci beaucoup"
        - "ok", "d'accord", "très bien", "parfait", "super", "cool", "bien", "génial"
        - "au revoir", "à plus", "salut", "bye", "ciao", "bonne journée"
        - "c'est tout", "c'est bon", "ça marche", "j'ai compris", "noté"
        - "non", "non merci", "ça ira", "pas besoin"
        - "ah ok", "je vois", "entendu", "compris"
        
        NE PAS appeler end_conversation si l'utilisateur pose une question ou fait une demande.
        
        Si l'utilisateur mentionne ton nom ("{wake_word_name}") pendant la conversation, c'est normal."""
        
        # Speech recognition settings
        self.silence_timeout = silence_timeout  # Seconds of silence before considering speech ended
        self.conversation_timeout = conversation_timeout  # Seconds of no speech before ending conversation
        self.last_transcript_time = None
    
    def _get_greeting_filename(self, greeting_text):
        """Generate a safe filename for a greeting text."""
        import hashlib
        # Use hash to create unique, safe filename
        text_hash = hashlib.md5(greeting_text.encode('utf-8')).hexdigest()[:12]
        # Also include voice name to regenerate if voice changes
        return f"greeting_{self.tts_voice}_{text_hash}.mp3"
    
    def _ensure_greetings_cached(self):
        """Ensure all greetings are cached as audio files."""
        # Create cache directory if it doesn't exist
        os.makedirs(self.greetings_cache_dir, exist_ok=True)
        
        print("🔊 Vérification du cache des salutations...")
        
        for greeting in WAKE_WORD_GREETINGS:
            filename = self._get_greeting_filename(greeting)
            filepath = os.path.join(self.greetings_cache_dir, filename)
            
            if os.path.exists(filepath):
                # File already cached
                self._greeting_files[greeting] = filepath
                print(f"   ✓ Salutation en cache: \"{greeting}\"")
            else:
                # Need to generate TTS
                print(f"   🎤 Génération TTS pour: \"{greeting}\"...")
                try:
                    response = self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice=self.tts_voice,
                        input=greeting
                    )
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    self._greeting_files[greeting] = filepath
                    print(f"   ✓ Salutation générée et mise en cache: \"{greeting}\"")
                    
                except Exception as e:
                    print(f"   ❌ Erreur lors de la génération de \"{greeting}\": {e}")
        
        print(f"📁 Cache des salutations: {self.greetings_cache_dir}")
        print(f"   {len(self._greeting_files)}/{len(WAKE_WORD_GREETINGS)} salutations prêtes\n")
    
    def _ensure_fillers_cached(self):
        """Ensure all filler phrases are cached as audio files."""
        # Use the same cache directory as greetings
        os.makedirs(self.greetings_cache_dir, exist_ok=True)
        
        print("🔊 Vérification du cache des phrases de remplissage...")
        
        for filler in FILLER_PHRASES:
            filename = self._get_filler_filename(filler)
            filepath = os.path.join(self.greetings_cache_dir, filename)
            
            if os.path.exists(filepath):
                # File already cached
                self._filler_files[filler] = filepath
                print(f"   ✓ Filler en cache: \"{filler}\"")
            else:
                # Need to generate TTS
                print(f"   🎤 Génération TTS pour: \"{filler}\"...")
                try:
                    response = self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice=self.tts_voice,
                        input=filler
                    )
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    self._filler_files[filler] = filepath
                    print(f"   ✓ Filler généré et mis en cache: \"{filler}\"")
                    
                except Exception as e:
                    print(f"   ❌ Erreur lors de la génération de \"{filler}\": {e}")
        
        print(f"   {len(self._filler_files)}/{len(FILLER_PHRASES)} fillers prêts\n")
    
    def _get_filler_filename(self, filler_text):
        """Generate a safe filename for a filler phrase."""
        import hashlib
        text_hash = hashlib.md5(filler_text.encode('utf-8')).hexdigest()[:12]
        return f"filler_{self.tts_voice}_{text_hash}.mp3"
    
    def _play_filler_async(self):
        """Play a random filler phrase asynchronously. Returns the thread so caller can wait for it."""
        available_fillers = [f for f in FILLER_PHRASES if f in self._filler_files]
        
        if not available_fillers:
            return None
        
        filler = random.choice(available_fillers)
        filepath = self._filler_files[filler]
        
        print(f"   💭 \"{filler}\"")
        
        def play_filler():
            with self._audio_lock:
                self._is_speaking = True
                try:
                    self._play_headset_wakeup()
                    self._play_audio_file(filepath)
                finally:
                    self._is_speaking = False
        
        thread = threading.Thread(target=play_filler, daemon=True)
        thread.start()
        return thread
        
    def initialize_porcupine(self):
        """Initialize Porcupine with a wake word."""
        if self.custom_wake_word_path:
            # Use custom wake word file
            import pvporcupine
            
            # Determine model path
            model_path = None
            if self.porcupine_model_path and os.path.exists(self.porcupine_model_path):
                # Use configured model path
                model_path = self.porcupine_model_path
                print(f"Utilisation du modèle configuré: {model_path}")
            else:
                # Try to find French model in package directory
                porcupine_dir = os.path.dirname(pvporcupine.__file__)
                french_model_path = os.path.join(porcupine_dir, "lib", "common", "porcupine_params_fr.pv")
                
                if os.path.exists(french_model_path):
                    model_path = french_model_path
                    print(f"Utilisation du modèle français: {model_path}")
                else:
                    print(f"⚠️ Modèle français non trouvé à: {french_model_path}")
                    print(f"Tentative sans spécifier le modèle...")
            
            if model_path:
                self.porcupine = pvporcupine.create(
                    access_key=self.porcupine_access_key,
                    keyword_paths=[self.custom_wake_word_path],
                    model_path=model_path
                )
            else:
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
            print(f"(Dites '{self.wake_word}' pour démarrer une conversation)\n")
            
            while True:
                pcm = stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print("✓ Mot d'activation détecté!")
                    stream.stop_stream()
                    stream.close()
                    
                    # Start a conversation
                    self._run_conversation()
                    
                    # Restart stream for next wake word
                    stream = audio.open(
                        rate=self.porcupine.sample_rate,
                        channels=self.channels,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.porcupine.frame_length
                    )
                    print(f"\n🎤 Écoute du mot d'activation '{self.wake_word}'...\n")
                    
        except KeyboardInterrupt:
            print("\nArrêt...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
    
    def _clean_old_history(self):
        """Remove old history items based on age and count limits. Only called between conversations."""
        current_time = time.time()
        
        # Remove items older than max age
        self.conversation_history = [
            item for item in self.conversation_history
            if current_time - item["timestamp"] <= self.history_max_age_seconds
        ]
        
        # If still over max items, keep only the most recent ones
        if len(self.conversation_history) > self.max_history_items:
            self.conversation_history = self.conversation_history[-self.max_history_items:]
        
        print(f"   📜 Historique: {len(self.conversation_history)} messages conservés")
    
    def _add_to_history(self, message: dict):
        """Add a message to conversation history with timestamp."""
        self.conversation_history.append({
            "message": message,
            "timestamp": time.time()
        })
    
    def _get_messages_for_api(self) -> list:
        """Get messages formatted for the OpenAI API (without timestamps)."""
        return [item["message"] for item in self.conversation_history]
    
    def _run_conversation(self):
        """Run a conversation until it naturally ends."""
        # Clean old history items before starting new conversation (not during)
        self._clean_old_history()
        
        # Reset conversation state
        self._conversation_active = True
        self._conversation_should_end = False
        
        # Greet the user
        self.play_acknowledgment_greeting()
        
        print("\n💬 Conversation démarrée")
        print("   (Dites 'merci', 'au revoir', etc. pour terminer)\n")
        
        # First turn - listen for initial command
        self.listen_for_command()
        
        # Continue conversation until it should end
        while self._conversation_active and not self._conversation_should_end:
            # Play a soft beep to indicate we're still listening
            self.play_listening_beep()
            
            # Listen for next input
            print("🎙️ Je vous écoute...")
            self.listen_for_command()
        
        # Conversation ended
        self.play_end_conversation_sound()
        print("👋 Conversation terminée - Retour au mode mot d'activation\n")
        self._conversation_active = False
        self._boost_mode = False  # Reset boost mode for next conversation
    
    def play_acknowledgment_greeting(self):
        """Play a random voice greeting to acknowledge wake word detection."""
        with self._audio_lock:
            self._is_speaking = True
            try:
                # Choose a random greeting
                available_greetings = [g for g in WAKE_WORD_GREETINGS if g in self._greeting_files]
                
                if not available_greetings:
                    # Fallback to beep if no greetings available
                    print("⚠️ Aucune salutation disponible, utilisation du bip")
                    self._play_fallback_beep()
                    return
                
                greeting = random.choice(available_greetings)
                filepath = self._greeting_files[greeting]
                
                print(f"🗣️ \"{greeting}\"")
                
                # Wake up wireless headset
                self._play_headset_wakeup()
                
                # Play the cached greeting
                self._play_audio_file(filepath)
                
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de la salutation: {e}")
                self._play_fallback_beep()
            finally:
                self._is_speaking = False
    
    def _play_fallback_beep(self):
        """Play a simple beep as fallback if voice greeting fails."""
        try:
            import pygame
            
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
            
        except Exception as e:
            print("🔔 Beep!")
    
    def play_end_conversation_sound(self):
        """Play a descending tone to indicate conversation has ended."""
        with self._audio_lock:
            try:
                import pygame
                
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
    
    def play_listening_beep(self):
        """Play a soft, short beep to indicate the assistant is listening."""
        with self._audio_lock:
            try:
                import pygame
                
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
        print("🎙️  En écoute... (parlez maintenant)")
        
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
        self._got_any_speech = False  # Track if we got any speech
        audio_thread = threading.Thread(target=self._audio_stream_generator, args=(stream,))
        audio_thread.start()
        
        # Start conversation timeout checker (only during active conversation)
        start_time = time.time()
        timeout_triggered = [False]
        
        def conversation_timeout_checker():
            """Check for conversation timeout when no speech at all."""
            while self.listening_for_command:
                if self._conversation_active and not self._got_any_speech:
                    elapsed = time.time() - start_time
                    if elapsed > self.conversation_timeout:
                        print(f"\n   [Pas de réponse - fin de conversation]")
                        timeout_triggered[0] = True
                        self.listening_for_command = False
                        self.audio_queue.put(None)  # Sentinel to stop audio generator
                        break
                time.sleep(0.1)
        
        timeout_thread = None
        if self._conversation_active:
            timeout_thread = threading.Thread(target=conversation_timeout_checker, daemon=True)
            timeout_thread.start()
        
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
                print("❌ Aucun discours détecté.")
                self._conversation_should_end = True
                
        except Exception as e:
            # Don't print error if it was just a timeout
            if not timeout_triggered[0]:
                print(f"❌ Erreur lors de la reconnaissance vocale: {e}")
            self._conversation_should_end = True
        finally:
            self.listening_for_command = False
            if timeout_thread:
                timeout_thread.join(timeout=0.5)
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
        last_activity_time = [time.time()]
        should_stop = [False]
        got_any_speech = [False]
        
        def timeout_checker():
            """Check for silence timeout after speech has started."""
            while not should_stop[0]:
                current_time = time.time()
                
                # If we got speech, check for silence timeout
                if got_any_speech[0] and current_time - last_activity_time[0] > self.silence_timeout:
                    print(f"\n   [Silence détectée - fin de commande]")
                    self.listening_for_command = False
                    self.audio_queue.put(None)  # Sentinel to stop audio generator
                    should_stop[0] = True
                    break
                    
                time.sleep(0.1)
        
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
                
                got_any_speech[0] = True
                self._got_any_speech = True  # Also set instance variable for conversation timeout
                last_activity_time[0] = time.time()
                    
                if not result.is_final:
                    last_interim_transcript = result.alternatives[0].transcript
                    # Truncate to console width to avoid wrapping issues
                    try:
                        console_width = os.get_terminal_size().columns
                    except:
                        console_width = 80
                    prefix = "   (provisoire: "
                    suffix = "...)"
                    max_text_len = console_width - len(prefix) - len(suffix) - 1
                    if len(last_interim_transcript) > max_text_len:
                        display_text = prefix + last_interim_transcript[:max_text_len] + suffix
                    else:
                        display_text = prefix + last_interim_transcript + ")"
                    # Pad with spaces to clear previous longer text
                    display_text = display_text.ljust(console_width - 1)
                    print(display_text, end='\r')
                else:
                    last_final_transcript = result.alternatives[0].transcript
                    transcript = last_final_transcript
                    print(f"\n   [Confirmé: {transcript}]")
                    
        finally:
            should_stop[0] = True
            timeout_thread.join(timeout=0.5)
        
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
        
        # Reset filler flag for this interaction
        self._filler_played_this_interaction = False
        
        # Check for end phrases locally as a fallback
        end_phrases = [
            "merci", "ok merci", "merci beaucoup", "thanks",
            "au revoir", "à plus", "salut", "bye", "ciao", "bonne journée", "à bientôt",
            "c'est tout", "c'est bon", "ça marche", "j'ai compris", "noté",
            "non merci", "ça ira", "pas besoin", "c'est pas grave",
            "parfait", "super", "génial", "cool", "nickel", "top",
            "d'accord", "ok d'accord", "très bien", "entendu",
        ]
        command_lower = command.lower().strip()
        
        # Check if the command is just an end phrase (short response)
        is_likely_end = any(command_lower == phrase or command_lower == phrase + "." for phrase in end_phrases)
        # Also check for short responses starting with these phrases
        if not is_likely_end and len(command_lower.split()) <= 3:
            is_likely_end = any(command_lower.startswith(phrase) for phrase in end_phrases)
        
        try:
            # Add user message to conversation history
            self._add_to_history({
                "role": "user",
                "content": command
            })
            
            # Loop until GPT gives a final response without tool calls
            max_iterations = 5
            iteration = 0
            assistant_message = None
            
            while iteration < max_iterations:
                iteration += 1
                
                messages = [{"role": "system", "content": self.system_prompt}] + self._get_messages_for_api()
                
                # Select model based on boost mode
                model = "gpt-4o" if self._boost_mode else "gpt-4o-mini"
                
                # First, try streaming response
                # We need to detect if it's a tool call or text response
                stream = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=EXTENDED_TOOL_FUNCTIONS,
                    tool_choice="auto",
                    max_tokens=150,
                    temperature=0.7,
                    stream=True
                )
                
                # Process the stream
                result = self._process_streaming_response(stream)
                
                # If it's a text response (no tool calls), we're done
                if result["type"] == "text":
                    assistant_message = result["content"]
                    break
                
                # If it's a tool call, handle it
                elif result["type"] == "tool_calls":
                    tool_calls = result["tool_calls"]
                    
                    # Play a filler phrase on first tool call of this interaction
                    # Only if at least one function in this batch should trigger a filler
                    filler_thread = None
                    should_play_filler = any(tc["name"] not in NO_FILLER_FUNCTIONS for tc in tool_calls)
                    
                    if should_play_filler and not self._filler_played_this_interaction:
                        self._filler_played_this_interaction = True
                        filler_thread = self._play_filler_async()
                    
                    # Build tool calls for history
                    tool_calls_for_history = []
                    for tc in tool_calls:
                        tool_calls_for_history.append({
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"]
                            }
                        })
                    
                    self._add_to_history({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_for_history
                    })
                    
                    # Execute each tool and add results
                    for tc in tool_calls:
                        function_name = tc["name"]
                        function_args = json.loads(tc["arguments"])
                        
                        print(f"🔧 Appel de fonction: {function_name}({function_args})")
                        
                        # Handle end_conversation function
                        if function_name == "end_conversation":
                            print(f"   → Fin de conversation demandée")
                            self._conversation_should_end = True
                            function_result = {"success": True, "message": "Conversation terminée"}
                        elif function_name == "enable_boost_mode":
                            print(f"   → Mode boost activé (gpt-4o)")
                            self._boost_mode = True
                            function_result = {"success": True, "message": "Mode intelligence augmentée activé. J'utilise maintenant un modèle plus performant."}
                        elif function_name == "disable_boost_mode":
                            print(f"   → Mode boost désactivé (gpt-4o-mini)")
                            self._boost_mode = False
                            function_result = {"success": True, "message": "Mode normal rétabli."}
                        else:
                            # Execute other tool functions
                            function_result = self._execute_tool(function_name, function_args)
                        
                        # Add tool result to conversation
                        self._add_to_history({
                            "role": "tool",
                            "content": json.dumps(function_result, ensure_ascii=False),
                            "tool_call_id": tc["id"]
                        })
                    
                    # Wait for filler to finish before continuing to next iteration
                    if filler_thread and filler_thread.is_alive():
                        filler_thread.join()
            
            # Add assistant response to history
            if assistant_message:
                self._add_to_history({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                print(f"💬 Assistant: {assistant_message}")
                
                # Note: speak was already called during streaming for text responses
                # Only speak here if streaming didn't handle it (shouldn't happen normally)
                
                # Fallback: if user said an end phrase and GPT didn't call end_conversation, end anyway
                if is_likely_end and not self._conversation_should_end:
                    print("   → Fin de conversation détectée localement")
                    self._conversation_should_end = True
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de la commande: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_streaming_response(self, stream):
        """
        Process a streaming response from GPT.
        
        Returns a dict with either:
        - {"type": "text", "content": "full text"} for text responses
        - {"type": "tool_calls", "tool_calls": [...]} for tool calls
        
        For text responses, TTS is started as sentences are detected.
        """
        import re
        
        # Accumulators
        full_content = ""
        current_buffer = ""
        tool_calls = {}  # id -> {id, name, arguments}
        
        # Track what we're receiving
        is_tool_call = False
        
        # TTS pipeline - two queues for parallel generation and playback
        sentence_queue = queue.Queue()  # Sentences waiting for TTS generation
        audio_queue = queue.Queue()     # Audio files ready to play
        tts_generator_thread = None
        audio_player_thread = None
        tts_started = False
        generation_done = threading.Event()
        
        def tts_generator_worker():
            """Background thread that generates TTS for sentences."""
            try:
                while True:
                    try:
                        sentence = sentence_queue.get(timeout=1)
                        if sentence is None:  # Poison pill
                            break
                        
                        print(f"   🎵 TTS: \"{sentence[:40]}{'...' if len(sentence) > 40 else ''}\"")
                        audio_path = self._generate_tts_audio(sentence)
                        audio_queue.put(audio_path)
                        
                    except queue.Empty:
                        # Check if we're done receiving sentences
                        if generation_done.is_set():
                            break
                        continue
            finally:
                audio_queue.put(None)  # Signal player that generation is done
        
        def audio_player_worker():
            """Background thread that plays audio files."""
            with self._audio_lock:
                self._is_speaking = True
                
                try:
                    first_sentence = True
                    while True:
                        try:
                            audio_path = audio_queue.get(timeout=30)
                            if audio_path is None:  # Poison pill
                                break
                            
                            # Wake up headset before first sentence
                            if first_sentence:
                                self._play_headset_wakeup()
                                first_sentence = False
                            else:
                                time.sleep(0.1)  # Small pause between sentences
                            
                            self._play_audio_file(audio_path)
                            
                            try:
                                os.unlink(audio_path)
                            except:
                                pass
                                
                        except queue.Empty:
                            break
                finally:
                    self._is_speaking = False
        
        def extract_complete_sentence(buffer):
            """
            Extract a complete sentence from buffer if available.
            Returns (sentence, remaining_buffer) or (None, buffer) if no complete sentence.
            
            Handles bullet numbers like "1." by not treating them as sentence ends.
            """
            if not buffer:
                return None, buffer
            
            # First, normalize bullet numbers: "1." or "1. " at start -> "1: "
            # But only if followed by more text
            normalized = re.sub(r'^(\d+)\.\s*', r'\1: ', buffer)
            
            # Look for sentence-ending punctuation
            # Match . ! ? followed by space or end of string
            # But not after a digit (to avoid "1." being treated as sentence end)
            match = re.search(r'(?<![0-9])([.!?])\s+', normalized)
            
            if match:
                # Found a sentence boundary
                end_pos = match.end()
                sentence = normalized[:match.start() + 1].strip()
                remaining = normalized[end_pos:]
                return sentence, remaining
            
            # Check if buffer ends with sentence-ending punctuation (final sentence)
            # But not if it's just a number followed by period
            if normalized and normalized[-1] in '.!?' and not re.match(r'^\d+\.$', normalized.strip()):
                return normalized.strip(), ""
            
            return None, buffer
        
        # Process the stream
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            
            if delta is None:
                continue
            
            # Check for tool calls
            if delta.tool_calls:
                is_tool_call = True
                for tc in delta.tool_calls:
                    tc_id = tc.id or list(tool_calls.keys())[-1] if tool_calls else None
                    
                    if tc.id:  # New tool call
                        tool_calls[tc.id] = {
                            "id": tc.id,
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                        }
                    elif tc_id:  # Continuation of existing tool call
                        if tc.function:
                            if tc.function.name:
                                tool_calls[tc_id]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[tc_id]["arguments"] += tc.function.arguments
            
            # Check for content
            if delta.content:
                content_chunk = delta.content
                full_content += content_chunk
                current_buffer += content_chunk
                
                # Start TTS threads on first content if not started
                if not tts_started and not is_tool_call:
                    tts_started = True
                    tts_generator_thread = threading.Thread(target=tts_generator_worker, daemon=True)
                    audio_player_thread = threading.Thread(target=audio_player_worker, daemon=True)
                    tts_generator_thread.start()
                    audio_player_thread.start()
                    print("🔊 Streaming TTS...")
                
                # Try to extract complete sentences
                while True:
                    sentence, current_buffer = extract_complete_sentence(current_buffer)
                    if sentence:
                        sentence_queue.put(sentence)
                    else:
                        break
        
        # Handle any remaining text in buffer
        if current_buffer.strip() and not is_tool_call:
            # Normalize any remaining bullet numbers
            remaining = re.sub(r'^(\d+)\.\s*', r'\1: ', current_buffer.strip())
            if remaining:
                sentence_queue.put(remaining)
        
        # Signal TTS generator that we're done sending sentences
        generation_done.set()
        sentence_queue.put(None)  # Poison pill for generator
        
        # Wait for threads to finish
        if tts_generator_thread:
            tts_generator_thread.join(timeout=60)
        if audio_player_thread:
            audio_player_thread.join(timeout=60)
        
        # Return appropriate result
        if is_tool_call:
            return {
                "type": "tool_calls",
                "tool_calls": list(tool_calls.values())
            }
        else:
            return {
                "type": "text",
                "content": full_content
            }
    
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
            elif function_name == "set_alarm":
                return self.tools.set_alarm(**function_args)
            elif function_name == "search_web":
                return self.tools.search_web(**function_args)
            elif function_name == "load_webpage":
                return self.tools.load_webpage(**function_args)
            elif function_name == "send_email":
                return self.tools.send_email(**function_args)
            elif function_name == "send_sms":
                return self.tools.send_sms(**function_args)
            elif function_name == "get_news":
                return self.tools.get_news(**function_args)
            elif function_name == "get_calendar_events":
                return self.tools.get_calendar_events(**function_args)
            elif function_name == "add_calendar_event":
                return self.tools.add_calendar_event(**function_args)
            elif function_name == "random_choice":
                return self.tools.random_choice(**function_args)
            elif function_name == "get_memories":
                return self.tools.get_memories()
            elif function_name == "save_memories":
                return self.tools.save_memories(**function_args)
            else:
                return {"success": False, "error": f"Fonction inconnue: {function_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences based on punctuation and newlines."""
        import re
        
        # First, join numbered list items with their content
        # Replace "1.\n" or "1. " at start of line with "1: " to prevent splitting
        text = re.sub(r'(\d+)\.\s*\n?', r'\1: ', text)
        
        # Split on sentence-ending punctuation followed by space or end, or on newlines
        # BUT don't split on colons after numbers (our numbered list format)
        sentences = re.split(r'(?<![0-9])(?<=[.!?])\s+|\n+', text)
        
        # Filter out empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _generate_tts_audio(self, text: str) -> str:
        """Generate TTS audio for text and return the temp file path."""
        # For very short texts that might be ambiguous (like numbers),
        # the TTS might not detect French properly. We don't modify the text
        # since OpenAI TTS doesn't have a language parameter, but the 
        # sentence splitting should prevent isolated numbers.
        response = self.openai_client.audio.speech.create(
            model="tts-1",
            voice=self.tts_voice,
            input=text
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            temp_audio.write(response.content)
            return temp_audio.name
    
    def speak(self, text):
        """Convert text to speech using OpenAI TTS with pipelined streaming for faster response."""
        print("🔊 Prononciation de la réponse...")
        
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return
        
        # If only one sentence, use simple approach
        if len(sentences) == 1:
            self._speak_simple(text)
            return
        
        print(f"   📝 Pipeline TTS: {len(sentences)} parties")
        
        with self._audio_lock:
            self._is_speaking = True
            
            try:
                # Wake up wireless headset
                self._play_headset_wakeup()
                
                # Queue to hold generated audio files ready to play
                audio_queue = queue.Queue()
                generation_complete = threading.Event()
                generation_error = [None]
                
                def generate_all_audio():
                    """Background thread that generates all TTS audio."""
                    try:
                        for i, sentence in enumerate(sentences):
                            print(f"   🎵 Génération partie {i+1}/{len(sentences)}: \"{sentence[:40]}{'...' if len(sentence) > 40 else ''}\"")
                            audio_path = self._generate_tts_audio(sentence)
                            audio_queue.put(audio_path)
                    except Exception as e:
                        generation_error[0] = e
                        audio_queue.put(None)  # Signal error
                    finally:
                        generation_complete.set()
                
                # Start background generation thread
                gen_thread = threading.Thread(target=generate_all_audio, daemon=True)
                gen_thread.start()
                
                # Play audio as it becomes available
                sentences_played = 0
                while sentences_played < len(sentences):
                    try:
                        # Wait for next audio file (with timeout to check for errors)
                        audio_path = audio_queue.get(timeout=30)
                        
                        if audio_path is None:
                            # Error occurred in generation
                            if generation_error[0]:
                                print(f"   ❌ Erreur génération: {generation_error[0]}")
                            break
                        
                        # Small pause between sentences for natural flow (except before first)
                        if sentences_played > 0:
                            time.sleep(0.1)
                        
                        # Play this sentence
                        self._play_audio_file(audio_path)
                        sentences_played += 1
                        
                        # Clean up audio file
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
                            
                    except queue.Empty:
                        print("   ⚠️ Timeout en attente de l'audio")
                        break
                
                # Wait for generation thread to finish
                gen_thread.join(timeout=1)
                
            except Exception as e:
                print(f"❌ Erreur lors de la génération/lecture de la parole: {e}")
            finally:
                self._is_speaking = False
    
    def _speak_simple(self, text):
        """Simple TTS without chunking for short texts."""
        with self._audio_lock:
            self._is_speaking = True
            
            try:
                # Wake up wireless headset
                self._play_headset_wakeup()
                
                # Generate speech using OpenAI TTS
                response = self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice=self.tts_voice,
                    input=text
                )
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                    temp_audio.write(response.content)
                    temp_audio_path = temp_audio.name
                
                # Play the audio file
                self._play_audio_file(temp_audio_path)
                
                # Clean up
                time.sleep(0.5)
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                
            except Exception as e:
                print(f"❌ Erreur lors de la génération/lecture de la parole: {e}")
            finally:
                self._is_speaking = False
            
    def _play_headset_wakeup(self):
        """Play a very brief, nearly silent tone to wake up wireless headsets."""
        try:
            import pygame
            
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            sample_rate = 22050
            duration = 0.05
            frequency = 440
            
            samples = int(sample_rate * duration)
            wave_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
            wave_data = wave_data * 0.01
            wave_data = (wave_data * 32767).astype(np.int16)
            stereo_data = np.column_stack((wave_data, wave_data))
            
            sound = pygame.sndarray.make_sound(stereo_data)
            sound.play()
            
            while pygame.mixer.get_busy():
                pygame.time.wait(10)
            
            pygame.mixer.quit()
            
        except Exception:
            pass
    
    def _play_audio_file(self, file_path):
        """Play an audio file using pygame."""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
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
        print("\n📋 Mode de fonctionnement:")
        print(f"   • Dites '{self.wake_word}' pour démarrer une conversation")
        print("   • La conversation continue jusqu'à sa fin naturelle")
        print("   • Dites 'merci', 'au revoir', etc. pour terminer\n")
        
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
        TWILIO_SMS_FROM,
        SMS_CONTACTS,
        RSS_FEEDS
    )
    
    # Optional email configuration
    try:
        from config import EMAIL_SMTP_SERVER
    except ImportError:
        EMAIL_SMTP_SERVER = None
    
    try:
        from config import EMAIL_SMTP_PORT
    except ImportError:
        EMAIL_SMTP_PORT = 587
    
    try:
        from config import EMAIL_ADDRESS
    except ImportError:
        EMAIL_ADDRESS = None
    
    try:
        from config import EMAIL_PASSWORD
    except ImportError:
        EMAIL_PASSWORD = None
    
    try:
        from config import EMAIL_CONTACTS
    except ImportError:
        EMAIL_CONTACTS = {}
    
    # Optional imports
    try:
        from config import BEEP_VOLUME
    except ImportError:
        BEEP_VOLUME = 0.3
    
    try:
        from config import GOOGLE_CALENDAR_CREDENTIALS_PATH
    except ImportError:
        GOOGLE_CALENDAR_CREDENTIALS_PATH = None
    
    try:
        from config import GOOGLE_CALENDAR_ID
    except ImportError:
        GOOGLE_CALENDAR_ID = None
    
    try:
        from config import GREETINGS_CACHE_DIR
    except ImportError:
        GREETINGS_CACHE_DIR = None
    
    try:
        from config import PORCUPINE_MODEL_PATH
    except ImportError:
        PORCUPINE_MODEL_PATH = None
    
    # Optional: history configuration
    try:
        from config import MAX_HISTORY_ITEMS
    except ImportError:
        MAX_HISTORY_ITEMS = 50  # Default: keep up to 50 messages
    
    try:
        from config import HISTORY_MAX_AGE_SECONDS
    except ImportError:
        HISTORY_MAX_AGE_SECONDS = 3600  # Default: 1 hour
    
    try:
        from config import MEMORY_FILE_PATH
    except ImportError:
        MEMORY_FILE_PATH = None  # Default: ~/.casimir_memory.txt
    
    try:
        from config import DEFAULT_CITY
    except ImportError:
        DEFAULT_CITY = None  # No default city
    
    try:
        from config import SILENCE_TIMEOUT
    except ImportError:
        SILENCE_TIMEOUT = 1.5  # Default: 1.5 seconds of silence before speech is considered ended
    
    try:
        from config import CONVERSATION_TIMEOUT
    except ImportError:
        CONVERSATION_TIMEOUT = 10.0  # Default: 10 seconds of no speech before ending conversation
    
    # Create and run the assistant
    assistant = VoiceAssistant(
        porcupine_access_key=PORCUPINE_ACCESS_KEY,
        openai_api_key=OPENAI_API_KEY,
        google_credentials_path=GOOGLE_CREDENTIALS_PATH,
        openweather_api_key=OPENWEATHER_API_KEY,
        tts_voice=TTS_VOICE,
        wake_word=WAKE_WORD,
        custom_wake_word_path=CUSTOM_WAKE_WORD_PATH,
        porcupine_model_path=PORCUPINE_MODEL_PATH,
        google_search_api_key=GOOGLE_SEARCH_API_KEY,
        google_search_engine_id=GOOGLE_SEARCH_ENGINE_ID,
        twilio_account_sid=TWILIO_ACCOUNT_SID,
        twilio_auth_token=TWILIO_AUTH_TOKEN,
        twilio_sms_from=TWILIO_SMS_FROM,
        sms_contacts=SMS_CONTACTS,
        rss_feeds=RSS_FEEDS,
        beep_volume=BEEP_VOLUME,
        google_calendar_credentials_path=GOOGLE_CALENDAR_CREDENTIALS_PATH,
        google_calendar_id=GOOGLE_CALENDAR_ID,
        greetings_cache_dir=GREETINGS_CACHE_DIR,
        max_history_items=MAX_HISTORY_ITEMS,
        history_max_age_seconds=HISTORY_MAX_AGE_SECONDS,
        memory_file_path=MEMORY_FILE_PATH,
        default_city=DEFAULT_CITY,
        email_smtp_server=EMAIL_SMTP_SERVER,
        email_smtp_port=EMAIL_SMTP_PORT,
        email_address=EMAIL_ADDRESS,
        email_password=EMAIL_PASSWORD,
        email_contacts=EMAIL_CONTACTS,
        silence_timeout=SILENCE_TIMEOUT,
        conversation_timeout=CONVERSATION_TIMEOUT
    )
    
    assistant.run()
