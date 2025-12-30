import datetime
import requests
import json
import os
from typing import Dict, Any, Optional

class AssistantTools:
    """Tools/functions that the assistant can use to fetch real-time information."""
    
    def __init__(self, openweather_api_key=None, google_search_api_key=None, google_search_engine_id=None, twilio_account_sid=None, twilio_auth_token=None, twilio_whatsapp_from=None, whatsapp_contacts=None, twilio_sms_from=None, sms_contacts=None, rss_feeds=None, google_calendar_credentials_path=None, google_calendar_id=None, memory_file_path=None):
        """
        Initialize tools with necessary API keys.
        
        Args:
            openweather_api_key: API key for OpenWeatherMap (optional)
            google_search_api_key: API key for Google Custom Search (optional)
            google_search_engine_id: Search Engine ID for Google Custom Search (optional)
            twilio_account_sid: Twilio Account SID (optional)
            twilio_auth_token: Twilio Auth Token (optional)
            twilio_whatsapp_from: Twilio WhatsApp number (optional)
            whatsapp_contacts: Dictionary of contact names to WhatsApp numbers (optional)
            twilio_sms_from: Twilio phone number for SMS (optional)
            sms_contacts: Dictionary of contact names to phone numbers (optional)
            rss_feeds: Dictionary of category names to RSS feed URLs (optional)
            google_calendar_credentials_path: Path to Google OAuth credentials JSON file (optional)
            google_calendar_id: Google Calendar ID to use (optional, defaults to 'primary')
            memory_file_path: Path to memory file (optional, defaults to ~/.casimir_memory.txt)
        """
        self.openweather_api_key = openweather_api_key
        self.google_search_api_key = google_search_api_key
        self.google_search_engine_id = google_search_engine_id
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_whatsapp_from = twilio_whatsapp_from
        self.whatsapp_contacts = whatsapp_contacts or {}
        self.twilio_sms_from = twilio_sms_from
        self.sms_contacts = sms_contacts or {}
        self.rss_feeds = rss_feeds or {}
        self.google_calendar_credentials_path = google_calendar_credentials_path
        self.google_calendar_id = google_calendar_id or 'primary'
        self._calendar_service = None
        self.memory_file_path = memory_file_path or os.path.expanduser("~/.casimir_memory.txt")
        
    def _get_calendar_service(self):
        """Get or create the Google Calendar service."""
        if self._calendar_service is not None:
            return self._calendar_service
            
        if not self.google_calendar_credentials_path:
            return None
            
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import pickle
            
            SCOPES = ['https://www.googleapis.com/auth/calendar']
            
            creds = None
            token_path = os.path.join(os.path.dirname(self.google_calendar_credentials_path), 'token.pickle')
            
            # Load existing credentials
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.google_calendar_credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next time
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
            
            self._calendar_service = build('calendar', 'v3', credentials=creds)
            return self._calendar_service
            
        except Exception as e:
            print(f"[DEBUG Calendar] Error initializing service: {e}")
            return None
    
    def get_calendar_events(self, date: str = "today", days: int = 1) -> Dict[str, Any]:
        """
        Get calendar events for a specific date or date range.
        
        Args:
            date: Date to query - "today", "tomorrow", "yesterday", or ISO format (YYYY-MM-DD)
            days: Number of days to include (default 1)
            
        Returns:
            Dictionary with calendar events
        """
        print(f"[DEBUG Calendar] Getting events for date={date}, days={days}")
        
        service = self._get_calendar_service()
        if not service:
            return {
                "success": False,
                "error": "Google Calendar non configuré. Ajoutez GOOGLE_CALENDAR_CREDENTIALS_PATH dans config.py"
            }
        
        try:
            import pytz
            tz = pytz.timezone('Europe/Paris')
            now = datetime.datetime.now(tz)
            
            # Parse the date
            if date.lower() == "today" or date.lower() == "aujourd'hui":
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif date.lower() == "tomorrow" or date.lower() == "demain":
                start_date = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            elif date.lower() == "yesterday" or date.lower() == "hier":
                start_date = (now - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # Try to parse ISO format
                try:
                    start_date = datetime.datetime.fromisoformat(date)
                    if start_date.tzinfo is None:
                        start_date = tz.localize(start_date)
                    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                except:
                    return {"success": False, "error": f"Format de date invalide: {date}"}
            
            end_date = start_date + datetime.timedelta(days=days)
            
            # Convert to RFC3339 format
            time_min = start_date.isoformat()
            time_max = end_date.isoformat()
            
            print(f"[DEBUG Calendar] Querying from {time_min} to {time_max}")
            
            events_result = service.events().list(
                calendarId=self.google_calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Parse and format times
                if 'T' in start:
                    start_dt = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                    start_formatted = start_dt.strftime('%H:%M')
                else:
                    start_formatted = "Toute la journée"
                
                if 'T' in end:
                    end_dt = datetime.datetime.fromisoformat(end.replace('Z', '+00:00'))
                    end_formatted = end_dt.strftime('%H:%M')
                else:
                    end_formatted = ""
                
                formatted_events.append({
                    "title": event.get('summary', 'Sans titre'),
                    "start": start_formatted,
                    "end": end_formatted,
                    "location": event.get('location', ''),
                    "description": event.get('description', '')
                })
            
            print(f"[DEBUG Calendar] Found {len(formatted_events)} events")
            
            return {
                "success": True,
                "date": start_date.strftime('%A %d %B %Y'),
                "days": days,
                "events": formatted_events,
                "total_events": len(formatted_events)
            }
            
        except Exception as e:
            print(f"[DEBUG Calendar] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Erreur lors de la récupération des événements: {str(e)}"
            }
    
    def add_calendar_event(self, title: str, date: str, start_time: str = None, end_time: str = None, description: str = "", location: str = "") -> Dict[str, Any]:
        """
        Add an event to the calendar.
        
        Args:
            title: Event title/summary
            date: Date for the event - "today", "tomorrow", or ISO format (YYYY-MM-DD)
            start_time: Start time in HH:MM format (optional, if not provided creates all-day event)
            end_time: End time in HH:MM format (optional, defaults to 1 hour after start)
            description: Event description (optional)
            location: Event location (optional)
            
        Returns:
            Dictionary with creation status
        """
        print(f"[DEBUG Calendar] Adding event: {title} on {date} at {start_time}")
        
        service = self._get_calendar_service()
        if not service:
            return {
                "success": False,
                "error": "Google Calendar non configuré. Ajoutez GOOGLE_CALENDAR_CREDENTIALS_PATH dans config.py"
            }
        
        try:
            import pytz
            tz = pytz.timezone('Europe/Paris')
            now = datetime.datetime.now(tz)
            
            # Parse the date
            if date.lower() == "today" or date.lower() == "aujourd'hui":
                event_date = now.date()
            elif date.lower() == "tomorrow" or date.lower() == "demain":
                event_date = (now + datetime.timedelta(days=1)).date()
            else:
                try:
                    event_date = datetime.datetime.fromisoformat(date).date()
                except:
                    return {"success": False, "error": f"Format de date invalide: {date}"}
            
            event = {
                'summary': title,
                'description': description,
                'location': location,
            }
            
            if start_time:
                # Timed event
                start_hour, start_minute = map(int, start_time.split(':'))
                start_dt = tz.localize(datetime.datetime.combine(event_date, datetime.time(start_hour, start_minute)))
                
                if end_time:
                    end_hour, end_minute = map(int, end_time.split(':'))
                    end_dt = tz.localize(datetime.datetime.combine(event_date, datetime.time(end_hour, end_minute)))
                else:
                    # Default to 1 hour duration
                    end_dt = start_dt + datetime.timedelta(hours=1)
                
                event['start'] = {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': 'Europe/Paris',
                }
                event['end'] = {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': 'Europe/Paris',
                }
            else:
                # All-day event
                event['start'] = {
                    'date': event_date.isoformat(),
                }
                event['end'] = {
                    'date': (event_date + datetime.timedelta(days=1)).isoformat(),
                }
            
            created_event = service.events().insert(calendarId=self.google_calendar_id, body=event).execute()
            
            print(f"[DEBUG Calendar] Event created: {created_event.get('id')}")
            
            return {
                "success": True,
                "message": f"Événement '{title}' créé",
                "event_id": created_event.get('id'),
                "title": title,
                "date": event_date.strftime('%A %d %B %Y'),
                "start_time": start_time or "Toute la journée",
                "end_time": end_time or ""
            }
            
        except Exception as e:
            print(f"[DEBUG Calendar] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Erreur lors de la création de l'événement: {str(e)}"
            }
        
    def get_current_time(self, timezone: str = "Europe/Paris") -> Dict[str, Any]:
        """
        Get the current time in a specific timezone.
        
        Args:
            timezone: Timezone string (e.g., "Europe/Paris", "America/New_York")
            
        Returns:
            Dictionary with current time information
        """
        try:
            import pytz
            tz = pytz.timezone(timezone)
            now = datetime.datetime.now(tz)
            
            return {
                "success": True,
                "time": now.strftime("%H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "day": now.strftime("%A"),
                "timezone": timezone,
                "formatted": now.strftime("%A %d %B %Y, %H:%M:%S")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_weather(self, city: str, country_code: str = "FR") -> Dict[str, Any]:
        """
        Get current weather for a city using OpenWeatherMap API.
        
        Args:
            city: City name (e.g., "Paris")
            country_code: ISO country code (e.g., "FR")
            
        Returns:
            Dictionary with weather information
        """
        if not self.openweather_api_key:
            return {
                "success": False,
                "error": "OpenWeatherMap API key not configured"
            }
        
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": f"{city},{country_code}",
                "appid": self.openweather_api_key,
                "units": "metric",
                "lang": "fr"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erreur API: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def calculate(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression (e.g., "2 + 2", "sqrt(16)")
            
        Returns:
            Dictionary with calculation result
        """
        try:
            # Using a safe evaluation method
            import math
            
            # Allowed functions and constants
            safe_dict = {
                'abs': abs, 'round': round, 'max': max, 'min': min,
                'sum': sum, 'pow': pow,
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                'tan': math.tan, 'log': math.log, 'log10': math.log10,
                'pi': math.pi, 'e': math.e
            }
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            return {
                "success": True,
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur de calcul: {str(e)}"
            }
    
    def set_timer(self, seconds: int, label: str = "") -> Dict[str, Any]:
        """
        Set a timer that will announce when complete.
        
        Args:
            seconds: Number of seconds for the timer
            label: Optional label for the timer
            
        Returns:
            Dictionary with timer information
        """
        import threading
        import time
        
        def timer_thread():
            """Background thread that waits and then announces timer completion."""
            time.sleep(seconds)
            
            # Wait if assistant is currently speaking
            while hasattr(self, '_is_speaking') and self._is_speaking:
                time.sleep(0.1)
            
            # Use assistant's audio lock to play alarm
            if hasattr(self, 'assistant') and hasattr(self.assistant, '_audio_lock'):
                with self.assistant._audio_lock:
                    self._play_timer_alarm()
            else:
                self._play_timer_alarm()
            
            # Announce timer completion
            if label:
                minutes = seconds // 60
                remaining_seconds = seconds % 60
                if minutes > 0 and remaining_seconds > 0:
                    time_str = f"{minutes} minute{'s' if minutes > 1 else ''} {remaining_seconds}"
                elif minutes > 0:
                    time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
                else:
                    time_str = f"{seconds} seconde{'s' if seconds > 1 else ''}"
                message = f"Votre timer de {time_str} pour {label} est terminé"
            else:
                minutes = seconds // 60
                remaining_seconds = seconds % 60
                if minutes > 0 and remaining_seconds > 0:
                    time_str = f"{minutes} minute{'s' if minutes > 1 else ''} {remaining_seconds}"
                elif minutes > 0:
                    time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
                else:
                    time_str = f"{seconds} seconde{'s' if seconds > 1 else ''}"
                message = f"Votre timer de {time_str} est terminé"
            
            # Store reference to parent assistant for speaking
            if hasattr(self, 'assistant'):
                self.assistant.speak(message)
        
        # Start timer in background thread
        thread = threading.Thread(target=timer_thread, daemon=True)
        thread.start()
        
        return {
            "success": True,
            "message": f"Timer de {seconds} secondes défini{' pour ' + label if label else ''}",
            "seconds": seconds,
            "label": label
        }
    
    def _play_timer_alarm(self):
        """Play an alarm sound for timer completion."""
        try:
            import pygame
            import numpy as np
            
            # Initialize pygame mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            
            # Generate a more urgent alarm sound (three short beeps)
            sample_rate = 22050
            beep_duration = 0.15  # seconds per beep
            pause_duration = 0.1  # seconds between beeps
            frequency = 880  # Hz (A5 note - higher pitch than wake word beep)
            
            # Generate three beeps
            all_beeps = []
            for i in range(3):
                # Generate sine wave for beep
                samples = int(sample_rate * beep_duration)
                wave_data = np.sin(2 * np.pi * frequency * np.linspace(0, beep_duration, samples))
                
                # Apply fade in/out
                fade_samples = int(sample_rate * 0.01)  # 10ms fade
                wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
                wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                all_beeps.append(wave_data)
                
                # Add pause (silence) between beeps (except after last beep)
                if i < 2:
                    pause_samples = int(sample_rate * pause_duration)
                    all_beeps.append(np.zeros(pause_samples))
            
            # Combine all beeps
            combined = np.concatenate(all_beeps)
            
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
            
        except Exception as e:
            print(f"⏰ [Alarm - sound failed: {e}]")
    
    def search_web(self, query: str) -> Dict[str, Any]:
        """
        Search the web using Google Custom Search API.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        if not self.google_search_api_key or not self.google_search_engine_id:
            return {
                "success": False,
                "error": "Google Search API non configuré. Ajoutez GOOGLE_SEARCH_API_KEY et GOOGLE_SEARCH_ENGINE_ID dans config.py"
            }
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_search_api_key,
                "cx": self.google_search_engine_id,
                "q": query,
                "num": 5  # Number of results (max 10)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information from results
            results = []
            if "items" in data:
                for item in data["items"]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erreur lors de la recherche: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def send_whatsapp_message(self, contact_name: str, message: str) -> Dict[str, Any]:
        """
        Send a WhatsApp message via Twilio.
        
        Args:
            contact_name: Name of the contact (must be in WHATSAPP_CONTACTS)
            message: Message to send
            
        Returns:
            Dictionary with send status
        """
        print(f"[DEBUG WhatsApp] Tentative d'envoi à '{contact_name}': {message}")
        
        if not self.twilio_account_sid or not self.twilio_auth_token:
            error_msg = "Twilio non configuré. Ajoutez TWILIO_ACCOUNT_SID et TWILIO_AUTH_TOKEN dans config.py"
            print(f"[DEBUG WhatsApp] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        print(f"[DEBUG WhatsApp] Credentials OK")
        
        # Normalize contact name (lowercase, strip spaces)
        contact_name = contact_name.lower().strip()
        print(f"[DEBUG WhatsApp] Contact normalisé: '{contact_name}'")
        print(f"[DEBUG WhatsApp] Contacts disponibles: {list(self.whatsapp_contacts.keys())}")
        
        # Check if contact exists
        if contact_name not in self.whatsapp_contacts:
            available = ", ".join(self.whatsapp_contacts.keys())
            error_msg = f"Contact '{contact_name}' non trouvé. Contacts disponibles: {available}"
            print(f"[DEBUG WhatsApp] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        to_number = self.whatsapp_contacts[contact_name]
        print(f"[DEBUG WhatsApp] Numéro de destination: {to_number}")
        print(f"[DEBUG WhatsApp] Numéro d'envoi: {self.twilio_whatsapp_from}")
        
        try:
            from twilio.rest import Client
            
            print(f"[DEBUG WhatsApp] Initialisation client Twilio...")
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            print(f"[DEBUG WhatsApp] Envoi du message...")
            twilio_message = client.messages.create(
                body=message,
                from_=self.twilio_whatsapp_from,
                to=to_number
            )
            
            print(f"[DEBUG WhatsApp] Message envoyé! SID: {twilio_message.sid}, Status: {twilio_message.status}")
            
            return {
                "success": True,
                "contact": contact_name,
                "message": message,
                "message_sid": twilio_message.sid,
                "status": twilio_message.status
            }
            
        except ImportError as e:
            error_msg = f"Twilio non installé: {e}"
            print(f"[DEBUG WhatsApp] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Erreur lors de l'envoi: {str(e)}"
            print(f"[DEBUG WhatsApp] ERREUR: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }
    
    def send_sms(self, contact_name: str, message: str) -> Dict[str, Any]:
        """
        Send an SMS via Twilio.
        
        Args:
            contact_name: Name of the contact (must be in SMS_CONTACTS)
            message: Message to send
            
        Returns:
            Dictionary with send status
        """
        print(f"[DEBUG SMS] Tentative d'envoi à '{contact_name}': {message}")
        
        if not self.twilio_account_sid or not self.twilio_auth_token:
            error_msg = "Twilio non configuré. Ajoutez TWILIO_ACCOUNT_SID et TWILIO_AUTH_TOKEN dans config.py"
            print(f"[DEBUG SMS] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        if not self.twilio_sms_from:
            error_msg = "Numéro SMS Twilio non configuré. Ajoutez TWILIO_SMS_FROM dans config.py"
            print(f"[DEBUG SMS] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        print(f"[DEBUG SMS] Credentials OK")
        
        # Normalize contact name (lowercase, strip spaces)
        contact_name = contact_name.lower().strip()
        print(f"[DEBUG SMS] Contact normalisé: '{contact_name}'")
        print(f"[DEBUG SMS] Contacts disponibles: {list(self.sms_contacts.keys())}")
        
        # Check if contact exists
        if contact_name not in self.sms_contacts:
            available = ", ".join(self.sms_contacts.keys())
            error_msg = f"Contact '{contact_name}' non trouvé. Contacts disponibles: {available}"
            print(f"[DEBUG SMS] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        to_number = self.sms_contacts[contact_name]
        print(f"[DEBUG SMS] Numéro de destination: {to_number}")
        print(f"[DEBUG SMS] Numéro d'envoi: {self.twilio_sms_from}")
        
        try:
            from twilio.rest import Client
            
            print(f"[DEBUG SMS] Initialisation client Twilio...")
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            print(f"[DEBUG SMS] Envoi du message...")
            twilio_message = client.messages.create(
                body=message,
                from_=self.twilio_sms_from,
                to=to_number
            )
            
            print(f"[DEBUG SMS] Message envoyé! SID: {twilio_message.sid}, Status: {twilio_message.status}")
            
            return {
                "success": True,
                "contact": contact_name,
                "message": message,
                "message_sid": twilio_message.sid,
                "status": twilio_message.status
            }
            
        except ImportError as e:
            error_msg = f"Twilio non installé: {e}"
            print(f"[DEBUG SMS] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Erreur lors de l'envoi: {str(e)}"
            print(f"[DEBUG SMS] ERREUR: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_news(self, category: str = "général") -> Dict[str, Any]:
        """
        Get latest news headlines from RSS feeds.
        
        Args:
            category: Category of news (général, france, monde, sport, tech, économie)
            
        Returns:
            Dictionary with news articles (titles only for brevity)
        """
        print(f"[DEBUG News] Récupération des actualités - category={category}")
        
        if not self.rss_feeds:
            error_msg = "RSS feeds non configurés"
            print(f"[DEBUG News] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        # Normalize category name
        category = category.lower().strip()
        
        # Check if category exists
        if category not in self.rss_feeds:
            available = ", ".join(self.rss_feeds.keys())
            error_msg = f"Catégorie '{category}' non trouvée. Catégories disponibles: {available}"
            print(f"[DEBUG News] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        feed_url = self.rss_feeds[category]
        print(f"[DEBUG News] URL du flux RSS: {feed_url}")
        
        try:
            import feedparser
            
            print(f"[DEBUG News] Parsing du flux RSS...")
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                print(f"[DEBUG News] Warning: Flux RSS malformé")
            
            print(f"[DEBUG News] Nombre d'entrées: {len(feed.entries)}")
            
            # Only take first 3 articles for brevity (was 5)
            articles = []
            for entry in feed.entries[:3]:
                articles.append({
                    "title": entry.get("title", ""),
                })
            
            print(f"[DEBUG News] Articles récupérés: {len(articles)}")
            if articles:
                print(f"[DEBUG News] Premier titre: {articles[0]['title'][:50]}...")
            
            return {
                "success": True,
                "category": category,
                "articles": articles,
                "total_results": len(articles),
                "instruction": "Présente ces titres de façon concise, en une phrase par titre maximum."
            }
            
        except ImportError:
            error_msg = "feedparser non installé. Installez avec: pip install feedparser"
            print(f"[DEBUG News] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Erreur lors de la récupération des actualités: {str(e)}"
            print(f"[DEBUG News] ERREUR: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }
    
    def random_choice(self, mode: str = "dice", sides: int = 6, choices: list = None) -> Dict[str, Any]:
        """
        Make a random choice - roll dice, flip coin, or pick from options.
        
        Args:
            mode: "dice" for rolling dice, "choice" for picking from options, "coin" for coin flip, "number" for random number
            sides: Number of sides on the dice (default 6, only used in dice mode) or max number in number mode
            choices: List of options to choose from (only used in choice mode)
            
        Returns:
            Dictionary with the random result
        """
        import random
        
        print(f"[DEBUG Random] mode={mode}, sides={sides}, choices={choices}")
        
        try:
            if mode == "dice":
                if sides < 2:
                    return {
                        "success": False,
                        "error": "Un dé doit avoir au moins 2 faces"
                    }
                if sides > 1000:
                    return {
                        "success": False,
                        "error": "Maximum 1000 faces pour un dé"
                    }
                
                result = random.randint(1, sides)
                
                return {
                    "success": True,
                    "mode": "dice",
                    "sides": sides,
                    "result": result,
                    "message": f"Le dé à {sides} faces indique: {result}"
                }
            
            elif mode == "choice":
                if not choices or len(choices) < 2:
                    return {
                        "success": False,
                        "error": "Il faut au moins 2 choix pour faire une sélection aléatoire"
                    }
                
                result = random.choice(choices)
                
                return {
                    "success": True,
                    "mode": "choice",
                    "options": choices,
                    "result": result,
                    "message": f"Parmi {len(choices)} options, le choix est: {result}"
                }
            
            elif mode == "coin":
                result = random.choice(["pile", "face"])
                
                return {
                    "success": True,
                    "mode": "coin",
                    "result": result,
                    "message": f"La pièce tombe sur: {result}"
                }
            
            elif mode == "number":
                max_num = sides if sides > 1 else 100
                result = random.randint(1, max_num)
                
                return {
                    "success": True,
                    "mode": "number",
                    "max": max_num,
                    "result": result,
                    "message": f"Nombre aléatoire entre 1 et {max_num}: {result}"
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Mode inconnu: {mode}. Utilisez 'dice', 'choice', 'coin', ou 'number'"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur lors du tirage aléatoire: {str(e)}"
            }
    
    def get_memories(self) -> Dict[str, Any]:
        """
        Read all stored memories.
        
        Returns:
            Dictionary with memories content
        """
        print(f"[DEBUG Memory] Reading memories from {self.memory_file_path}")
        
        try:
            if not os.path.exists(self.memory_file_path):
                return {
                    "success": True,
                    "memories": [],
                    "message": "Aucun souvenir enregistré."
                }
            
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            memories = data.get("memories", [])
            
            if not memories:
                return {
                    "success": True,
                    "memories": [],
                    "message": "Aucun souvenir enregistré."
                }
            
            print(f"[DEBUG Memory] Read {len(memories)} memories")
            
            return {
                "success": True,
                "memories": memories,
                "count": len(memories),
                "message": f"{len(memories)} souvenir(s) récupéré(s)."
            }
            
        except json.JSONDecodeError as e:
            error_msg = f"Erreur de format du fichier mémoire: {str(e)}"
            print(f"[DEBUG Memory] ERROR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Erreur lors de la lecture des souvenirs: {str(e)}"
            print(f"[DEBUG Memory] ERROR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def save_memories(self, memories: list) -> Dict[str, Any]:
        """
        Save memories (replaces all existing memories).
        
        Args:
            memories: List of memory objects. Each object should have:
                      - "content": The memory text
                      - "created_at": (optional) ISO timestamp when first created
                      - "updated_at": (optional) ISO timestamp when last updated
                      New items without timestamps will have them added automatically.
            
        Returns:
            Dictionary with save status
        """
        print(f"[DEBUG Memory] Saving memories to {self.memory_file_path}")
        
        try:
            # Handle empty list (clearing all memories)
            if not memories:
                if os.path.exists(self.memory_file_path):
                    os.remove(self.memory_file_path)
                    print("[DEBUG Memory] Memory file deleted")
                return {
                    "success": True,
                    "message": "Tous les souvenirs ont été effacés."
                }
            
            now = datetime.datetime.now().isoformat()
            
            # Process each memory item
            processed_memories = []
            for item in memories:
                if isinstance(item, str):
                    # Simple string, create new memory object
                    processed_memories.append({
                        "content": item,
                        "created_at": now,
                        "updated_at": now
                    })
                elif isinstance(item, dict):
                    # Already a dict, ensure it has timestamps
                    memory = {
                        "content": item.get("content", ""),
                        "created_at": item.get("created_at", now),
                        "updated_at": now  # Always update the updated_at timestamp
                    }
                    processed_memories.append(memory)
            
            data = {"memories": processed_memories}
            
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"[DEBUG Memory] Saved {len(processed_memories)} memories")
            
            return {
                "success": True,
                "count": len(processed_memories),
                "message": f"{len(processed_memories)} souvenir(s) enregistré(s)."
            }
            
        except Exception as e:
            error_msg = f"Erreur lors de l'enregistrement des souvenirs: {str(e)}"
            print(f"[DEBUG Memory] ERROR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }


# Define the function schemas for OpenAI function calling
TOOL_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Obtenir l'heure actuelle dans un fuseau horaire spécifique",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Fuseau horaire (ex: 'Europe/Paris', 'America/New_York', 'Asia/Tokyo')",
                        "default": "Europe/Paris"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtenir la météo actuelle pour une ville",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Nom de la ville (ex: 'Paris', 'Lyon', 'Marseille')"
                    },
                    "country_code": {
                        "type": "string",
                        "description": "Code pays ISO (ex: 'FR', 'US', 'UK')",
                        "default": "FR"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Effectuer un calcul mathématique",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression mathématique à évaluer (ex: '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Définir un minuteur/timer. L'utilisateur peut spécifier le temps en minutes, secondes, ou les deux (ex: '2 minutes 30', '5 minutes', '30 secondes'). Peut inclure une étiquette optionnelle pour identifier le timer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Nombre TOTAL de secondes pour le timer (ex: pour '2 minutes 30', utiliser 150 secondes)"
                    },
                    "label": {
                        "type": "string",
                        "description": "Étiquette optionnelle pour le timer (ex: 'cuisson des pâtes', 'les oeufs', 'la pizza')"
                    }
                },
                "required": ["seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Rechercher des informations sur le web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_message",
            "description": "Envoyer un message WhatsApp à un contact. Le nom du contact doit correspondre exactement à un contact configuré.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "Nom du contact (ex: 'moi', 'marie', 'papa', 'maman'). Doit être en minuscules."
                    },
                    "message": {
                        "type": "string",
                        "description": "Le message à envoyer"
                    }
                },
                "required": ["contact_name", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Envoyer un SMS (message texte) à un contact. Le nom du contact doit correspondre exactement à un contact configuré.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "Nom du contact (ex: 'moi', 'david', 'papa', 'maman'). Doit être en minuscules."
                    },
                    "message": {
                        "type": "string",
                        "description": "Le message à envoyer"
                    }
                },
                "required": ["contact_name", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Obtenir les 3 derniers titres d'actualités françaises. Présenter de façon très concise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Catégorie d'actualités: général (défaut), france, monde, sport, tech, économie",
                        "enum": ["général", "france", "monde", "sport", "tech", "économie"],
                        "default": "général"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Obtenir les événements du calendrier Google pour une date ou période donnée. Permet de voir les rendez-vous, réunions et événements planifiés.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date à consulter: 'today'/'aujourd'hui', 'tomorrow'/'demain', 'yesterday'/'hier', ou format YYYY-MM-DD",
                        "default": "today"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Nombre de jours à inclure (défaut: 1)",
                        "default": 1
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_calendar_event",
            "description": "Ajouter un événement au calendrier Google. Peut créer des événements avec ou sans heure précise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Titre de l'événement"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date de l'événement: 'today'/'aujourd'hui', 'tomorrow'/'demain', ou format YYYY-MM-DD"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Heure de début au format HH:MM (ex: '14:30'). Si non spécifié, crée un événement toute la journée."
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Heure de fin au format HH:MM (ex: '15:30'). Si non spécifié, durée de 1 heure par défaut."
                    },
                    "description": {
                        "type": "string",
                        "description": "Description de l'événement (optionnel)"
                    },
                    "location": {
                        "type": "string",
                        "description": "Lieu de l'événement (optionnel)"
                    }
                },
                "required": ["title", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "random_choice",
            "description": "Faire un tirage aléatoire UNIQUEMENT quand l'utilisateur demande EXPLICITEMENT le hasard. Utilisé pour: 'lance un dé', 'pile ou face', 'tire au sort', 'au hasard', 'aléatoirement', 'randomly'. NE PAS utiliser si l'utilisateur demande de l'aide pour choisir, des conseils, ou une recommandation - dans ces cas, donner des conseils utiles à la place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Type de tirage: 'dice' (dé), 'coin' (pile ou face), 'choice' (choisir parmi options), 'number' (nombre aléatoire)",
                        "enum": ["dice", "coin", "choice", "number"],
                        "default": "dice"
                    },
                    "sides": {
                        "type": "integer",
                        "description": "Nombre de faces du dé (défaut: 6) ou nombre maximum pour le mode 'number'",
                        "default": 6
                    },
                    "choices": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Liste des options parmi lesquelles choisir (uniquement pour le mode 'choice')"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_memories",
            "description": "Lire les souvenirs enregistrés. Utiliser quand l'utilisateur demande ce que tu as retenu, ce dont tu te souviens, ou veut consulter ses notes/informations sauvegardées. Retourne une liste d'objets avec 'content', 'created_at' et 'updated_at'.",
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
            "name": "save_memories",
            "description": "Enregistrer des souvenirs. Utiliser quand l'utilisateur demande de retenir, mémoriser, noter ou sauvegarder une information. IMPORTANT: Toujours appeler get_memories d'abord, puis modifier la liste et la renvoyer complète. Pour ajouter: ajouter un nouvel objet {'content': '...'} à la liste. Pour modifier: mettre à jour le 'content' de l'objet existant (garder son 'created_at'). Pour supprimer: retirer l'objet de la liste. Pour tout effacer: envoyer une liste vide [].",
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Le contenu du souvenir"
                                },
                                "created_at": {
                                    "type": "string",
                                    "description": "Date de création ISO (conserver si existant)"
                                },
                                "updated_at": {
                                    "type": "string",
                                    "description": "Date de mise à jour ISO (sera mis à jour automatiquement)"
                                }
                            },
                            "required": ["content"]
                        },
                        "description": "Liste complète des souvenirs à sauvegarder"
                    }
                },
                "required": ["memories"]
            }
        }
    }
]
