import datetime
import requests
import json
import os
import threading
from typing import Dict, Any, Optional

class AssistantTools:
    """Tools/functions that the assistant can use to fetch real-time information."""
    
    def __init__(self, openweather_api_key=None, google_search_api_key=None, google_search_engine_id=None, twilio_account_sid=None, twilio_auth_token=None, twilio_sms_from=None, sms_contacts=None, rss_feeds=None, google_calendar_credentials_path=None, google_calendar_id=None, memory_file_path=None, email_smtp_server=None, email_smtp_port=None, email_address=None, email_password=None, email_contacts=None):
        """
        Initialize tools with necessary API keys.
        
        Args:
            openweather_api_key: API key for OpenWeatherMap (optional)
            google_search_api_key: API key for Google Custom Search (optional)
            google_search_engine_id: Search Engine ID for Google Custom Search (optional)
            twilio_account_sid: Twilio Account SID (optional)
            twilio_auth_token: Twilio Auth Token (optional)
            twilio_sms_from: Twilio phone number for SMS (optional)
            sms_contacts: Dictionary of contact names to phone numbers (optional)
            rss_feeds: Dictionary of category names to RSS feed URLs (optional)
            google_calendar_credentials_path: Path to Google OAuth credentials JSON file (optional)
            google_calendar_id: Google Calendar ID to use (optional, defaults to 'primary')
            memory_file_path: Path to memory file (optional, defaults to ~/.casimir_memory.txt)
            email_smtp_server: SMTP server address (optional, e.g., 'smtp.gmail.com')
            email_smtp_port: SMTP server port (optional, e.g., 587)
            email_address: Email address to send from (optional)
            email_password: Email password or app password (optional)
            email_contacts: Dictionary of contact names to email addresses (optional)
        """
        self.openweather_api_key = openweather_api_key
        self.google_search_api_key = google_search_api_key
        self.google_search_engine_id = google_search_engine_id
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_sms_from = twilio_sms_from
        self.sms_contacts = sms_contacts or {}
        self.rss_feeds = rss_feeds or {}
        self.google_calendar_credentials_path = google_calendar_credentials_path
        self.google_calendar_id = google_calendar_id or 'primary'
        self._calendar_service = None
        self.memory_file_path = memory_file_path or os.path.expanduser("~/.casimir_memory.txt")
        self.email_smtp_server = email_smtp_server
        self.email_smtp_port = email_smtp_port
        self.email_address = email_address
        self.email_password = email_password
        self.email_contacts = email_contacts or {}
        
        # Registry of scheduled items (timers, alarms, tasks)
        self._scheduled_items = {}  # id -> {type, label, trigger_time, thread, cancelled}
        self._next_scheduled_id = 1
        self._scheduled_lock = threading.Lock()
    
    def _register_scheduled_item(self, item_type: str, label: str, trigger_time: datetime.datetime, thread: threading.Thread) -> int:
        """Register a scheduled item and return its ID."""
        with self._scheduled_lock:
            item_id = self._next_scheduled_id
            self._next_scheduled_id += 1
            self._scheduled_items[item_id] = {
                "type": item_type,
                "label": label,
                "trigger_time": trigger_time,
                "thread": thread,
                "cancelled": False
            }
            return item_id
    
    def _unregister_scheduled_item(self, item_id: int):
        """Remove a scheduled item from registry."""
        with self._scheduled_lock:
            if item_id in self._scheduled_items:
                del self._scheduled_items[item_id]
    
    def _is_cancelled(self, item_id: int) -> bool:
        """Check if a scheduled item has been cancelled."""
        with self._scheduled_lock:
            if item_id in self._scheduled_items:
                return self._scheduled_items[item_id]["cancelled"]
            return True  # If not found, treat as cancelled
        
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
        import time
        import pytz
        
        tz = pytz.timezone('Europe/Paris')
        trigger_time = datetime.datetime.now(tz) + datetime.timedelta(seconds=seconds)
        
        # We need to create the thread first, then register, then start
        item_id_holder = [None]
        
        def timer_thread():
            """Background thread that waits and then announces timer completion."""
            time.sleep(seconds)
            
            # Check if cancelled
            if self._is_cancelled(item_id_holder[0]):
                self._unregister_scheduled_item(item_id_holder[0])
                return
            
            # Wait if assistant is currently speaking
            while hasattr(self, 'assistant') and self.assistant._is_speaking:
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
            
            # Unregister after completion
            self._unregister_scheduled_item(item_id_holder[0])
        
        # Start timer in background thread
        thread = threading.Thread(target=timer_thread, daemon=True)
        item_id = self._register_scheduled_item("timer", label or f"Timer {seconds}s", trigger_time, thread)
        item_id_holder[0] = item_id
        thread.start()
        
        return {
            "success": True,
            "message": f"Timer de {seconds} secondes défini{' pour ' + label if label else ''} (ID: {item_id})",
            "id": item_id,
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
    
    def set_alarm(self, time_str: str, date_str: str = None, label: str = "") -> Dict[str, Any]:
        """
        Set an alarm for a specific time (and optionally date).
        
        Args:
            time_str: Time in HH:MM format (24-hour)
            date_str: Optional date - "today", "tomorrow", "après-demain", or YYYY-MM-DD format
            label: Optional label/reminder message for the alarm
            
        Returns:
            Dictionary with alarm information
        """
        import time
        import pytz
        
        try:
            tz = pytz.timezone('Europe/Paris')
            now = datetime.datetime.now(tz)
            
            # Parse time
            try:
                hour, minute = map(int, time_str.split(':'))
            except:
                return {"success": False, "error": f"Format d'heure invalide: {time_str}. Utilisez HH:MM"}
            
            # Parse date
            if date_str is None or date_str.lower() in ["today", "aujourd'hui"]:
                alarm_date = now.date()
            elif date_str.lower() in ["tomorrow", "demain"]:
                alarm_date = (now + datetime.timedelta(days=1)).date()
            elif date_str.lower() in ["après-demain", "apres-demain", "après demain", "apres demain"]:
                alarm_date = (now + datetime.timedelta(days=2)).date()
            else:
                try:
                    alarm_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                except:
                    return {"success": False, "error": f"Format de date invalide: {date_str}. Utilisez YYYY-MM-DD"}
            
            # Create target datetime
            alarm_time = datetime.time(hour, minute)
            alarm_datetime = tz.localize(datetime.datetime.combine(alarm_date, alarm_time))
            
            # If time is in the past for today, move to tomorrow
            if alarm_datetime <= now and date_str is None:
                alarm_date = (now + datetime.timedelta(days=1)).date()
                alarm_datetime = tz.localize(datetime.datetime.combine(alarm_date, alarm_time))
            
            # Calculate seconds until alarm
            seconds_until_alarm = (alarm_datetime - now).total_seconds()
            
            if seconds_until_alarm <= 0:
                return {"success": False, "error": "L'heure de l'alarme est déjà passée"}
            
            # Format for display
            if alarm_date == now.date():
                date_display = "aujourd'hui"
            elif alarm_date == (now + datetime.timedelta(days=1)).date():
                date_display = "demain"
            else:
                date_display = alarm_date.strftime("%A %d %B")
            
            time_display = f"{hour:02d}:{minute:02d}"
            
            # We need to create the thread first, then register, then start
            item_id_holder = [None]
            
            def alarm_thread():
                """Background thread that waits and then announces alarm."""
                time.sleep(seconds_until_alarm)
                
                # Check if cancelled
                if self._is_cancelled(item_id_holder[0]):
                    self._unregister_scheduled_item(item_id_holder[0])
                    return
                
                # Wait if assistant is currently speaking
                while hasattr(self, 'assistant') and self.assistant._is_speaking:
                    time.sleep(0.1)
                
                # Use assistant's audio lock to play alarm
                if hasattr(self, 'assistant') and hasattr(self.assistant, '_audio_lock'):
                    with self.assistant._audio_lock:
                        self._play_timer_alarm()
                else:
                    self._play_timer_alarm()
                
                # Announce alarm
                if label:
                    message = f"Rappel: {label}"
                else:
                    message = f"C'est l'heure! Il est {time_display}"
                
                # Store reference to parent assistant for speaking
                if hasattr(self, 'assistant'):
                    self.assistant.speak(message)
                
                # Unregister after completion
                self._unregister_scheduled_item(item_id_holder[0])
            
            # Start alarm in background thread
            thread = threading.Thread(target=alarm_thread, daemon=True)
            item_id = self._register_scheduled_item("alarme", label or f"Alarme {time_display}", alarm_datetime, thread)
            item_id_holder[0] = item_id
            thread.start()
            
            # Build confirmation message
            if label:
                confirm_msg = f"Alarme définie pour {date_display} à {time_display}: {label} (ID: {item_id})"
            else:
                confirm_msg = f"Alarme définie pour {date_display} à {time_display} (ID: {item_id})"
            
            return {
                "success": True,
                "message": confirm_msg,
                "id": item_id,
                "time": time_display,
                "date": date_display,
                "label": label,
                "seconds_until_alarm": int(seconds_until_alarm)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Erreur: {str(e)}"}
    
    def schedule_task(self, task: str, time_str: str, date_str: str = None, speak_result: bool = False) -> Dict[str, Any]:
        """
        Schedule a task to be executed at a specific time.
        The task will be processed by GPT which can call functions like send_email, get_weather, etc.
        
        Args:
            task: Description of the task to execute (e.g., "Envoie un email à moi avec la météo du jour")
            time_str: Time in HH:MM format (24-hour)
            date_str: Optional date - "today", "tomorrow", "après-demain", or YYYY-MM-DD format
            speak_result: If True, speak the result of the task; if False, execute silently
            
        Returns:
            Dictionary with task information
        """
        import time
        import pytz
        
        try:
            tz = pytz.timezone('Europe/Paris')
            now = datetime.datetime.now(tz)
            
            # Parse time
            try:
                hour, minute = map(int, time_str.split(':'))
            except:
                return {"success": False, "error": f"Format d'heure invalide: {time_str}. Utilisez HH:MM"}
            
            # Parse date
            if date_str is None or date_str.lower() in ["today", "aujourd'hui"]:
                task_date = now.date()
            elif date_str.lower() in ["tomorrow", "demain"]:
                task_date = (now + datetime.timedelta(days=1)).date()
            elif date_str.lower() in ["après-demain", "apres-demain", "après demain", "apres demain"]:
                task_date = (now + datetime.timedelta(days=2)).date()
            else:
                try:
                    task_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                except:
                    return {"success": False, "error": f"Format de date invalide: {date_str}. Utilisez YYYY-MM-DD"}
            
            # Create target datetime
            task_time = datetime.time(hour, minute)
            task_datetime = tz.localize(datetime.datetime.combine(task_date, task_time))
            
            # If time is in the past for today, move to tomorrow
            if task_datetime <= now and date_str is None:
                task_date = (now + datetime.timedelta(days=1)).date()
                task_datetime = tz.localize(datetime.datetime.combine(task_date, task_time))
            
            # Calculate seconds until task
            seconds_until_task = (task_datetime - now).total_seconds()
            
            if seconds_until_task <= 0:
                return {"success": False, "error": "L'heure de la tâche est déjà passée"}
            
            # Format for display
            if task_date == now.date():
                date_display = "aujourd'hui"
            elif task_date == (now + datetime.timedelta(days=1)).date():
                date_display = "demain"
            else:
                date_display = task_date.strftime("%A %d %B")
            
            time_display = f"{hour:02d}:{minute:02d}"
            
            # We need to create the thread first, then register, then start
            item_id_holder = [None]
            
            def task_thread():
                """Background thread that waits and then executes the task."""
                time.sleep(seconds_until_task)
                
                # Check if cancelled
                if self._is_cancelled(item_id_holder[0]):
                    self._unregister_scheduled_item(item_id_holder[0])
                    return
                
                print(f"🤖 [Tâche planifiée] Exécution: {task}")
                
                # Execute the task through GPT
                if hasattr(self, 'assistant'):
                    try:
                        self.assistant.execute_background_task(task, speak_result=speak_result)
                    except Exception as e:
                        print(f"❌ [Tâche planifiée] Erreur: {e}")
                else:
                    print(f"❌ [Tâche planifiée] Pas d'assistant disponible")
                
                # Unregister after completion
                self._unregister_scheduled_item(item_id_holder[0])
            
            # Start task in background thread
            thread = threading.Thread(target=task_thread, daemon=True)
            item_id = self._register_scheduled_item("tâche", task[:50], task_datetime, thread)
            item_id_holder[0] = item_id
            thread.start()
            
            # Build confirmation message
            confirm_msg = f"Tâche planifiée pour {date_display} à {time_display}: {task} (ID: {item_id})"
            
            return {
                "success": True,
                "message": confirm_msg,
                "id": item_id,
                "time": time_display,
                "date": date_display,
                "task": task,
                "speak_result": speak_result
            }
            
        except Exception as e:
            return {"success": False, "error": f"Erreur: {str(e)}"}
    
    def schedule_task_delay(self, task: str, seconds: int, speak_result: bool = False) -> Dict[str, Any]:
        """
        Schedule a task to be executed after a delay.
        The task will be processed by GPT which can call functions like send_email, get_weather, etc.
        
        Args:
            task: Description of the task to execute (e.g., "Envoie un email à moi avec la météo du jour")
            seconds: Number of seconds to wait before executing
            speak_result: If True, speak the result of the task; if False, execute silently
            
        Returns:
            Dictionary with task information
        """
        import time
        import pytz
        
        tz = pytz.timezone('Europe/Paris')
        trigger_time = datetime.datetime.now(tz) + datetime.timedelta(seconds=seconds)
        
        # Format delay for display
        if seconds >= 3600:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            delay_display = f"{hours}h{mins:02d}" if mins else f"{hours}h"
        elif seconds >= 60:
            mins = seconds // 60
            secs = seconds % 60
            delay_display = f"{mins}min{secs:02d}" if secs else f"{mins}min"
        else:
            delay_display = f"{seconds}s"
        
        # We need to create the thread first, then register, then start
        item_id_holder = [None]
        
        def task_thread():
            """Background thread that waits and then executes the task."""
            time.sleep(seconds)
            
            # Check if cancelled
            if self._is_cancelled(item_id_holder[0]):
                self._unregister_scheduled_item(item_id_holder[0])
                return
            
            print(f"🤖 [Tâche planifiée] Exécution: {task}")
            
            # Execute the task through GPT
            if hasattr(self, 'assistant'):
                try:
                    self.assistant.execute_background_task(task, speak_result=speak_result)
                except Exception as e:
                    print(f"❌ [Tâche planifiée] Erreur: {e}")
            else:
                print(f"❌ [Tâche planifiée] Pas d'assistant disponible")
            
            # Unregister after completion
            self._unregister_scheduled_item(item_id_holder[0])
        
        # Start task in background thread
        thread = threading.Thread(target=task_thread, daemon=True)
        item_id = self._register_scheduled_item("tâche", task[:50], trigger_time, thread)
        item_id_holder[0] = item_id
        thread.start()
        
        # Build confirmation message
        confirm_msg = f"Tâche planifiée dans {delay_display}: {task} (ID: {item_id})"
        
        return {
            "success": True,
            "message": confirm_msg,
            "id": item_id,
            "delay": delay_display,
            "seconds": seconds,
            "task": task,
            "speak_result": speak_result
        }
    
    def list_scheduled(self) -> Dict[str, Any]:
        """
        List all pending scheduled items (timers, alarms, tasks).
        
        Returns:
            Dictionary with list of scheduled items
        """
        import pytz
        
        tz = pytz.timezone('Europe/Paris')
        now = datetime.datetime.now(tz)
        
        with self._scheduled_lock:
            items = []
            for item_id, item in self._scheduled_items.items():
                if not item["cancelled"]:
                    # Calculate time remaining
                    if item["trigger_time"].tzinfo is None:
                        trigger_time = tz.localize(item["trigger_time"])
                    else:
                        trigger_time = item["trigger_time"]
                    
                    remaining = trigger_time - now
                    remaining_seconds = max(0, remaining.total_seconds())
                    
                    if remaining_seconds > 3600:
                        remaining_str = f"{int(remaining_seconds // 3600)}h {int((remaining_seconds % 3600) // 60)}min"
                    elif remaining_seconds > 60:
                        remaining_str = f"{int(remaining_seconds // 60)}min"
                    else:
                        remaining_str = f"{int(remaining_seconds)}s"
                    
                    items.append({
                        "id": item_id,
                        "type": item["type"],
                        "label": item["label"],
                        "trigger_time": trigger_time.strftime("%H:%M"),
                        "remaining": remaining_str
                    })
            
            # Sort by trigger time
            items.sort(key=lambda x: x["id"])
            
            return {
                "success": True,
                "count": len(items),
                "items": items
            }
    
    def cancel_scheduled(self, item_id: int = None, search_term: str = None) -> Dict[str, Any]:
        """
        Cancel a scheduled item by ID or by searching its label.
        
        Args:
            item_id: ID of the item to cancel
            search_term: Search term to find item by label (cancels first match)
            
        Returns:
            Dictionary with cancellation status
        """
        with self._scheduled_lock:
            if item_id is not None:
                if item_id in self._scheduled_items:
                    item = self._scheduled_items[item_id]
                    if not item["cancelled"]:
                        item["cancelled"] = True
                        return {
                            "success": True,
                            "message": f"{item['type'].capitalize()} '{item['label']}' (ID: {item_id}) annulé"
                        }
                    else:
                        return {"success": False, "error": f"L'élément {item_id} est déjà annulé"}
                else:
                    return {"success": False, "error": f"Aucun élément trouvé avec l'ID {item_id}"}
            
            elif search_term is not None:
                search_lower = search_term.lower()
                for iid, item in self._scheduled_items.items():
                    if not item["cancelled"] and search_lower in item["label"].lower():
                        item["cancelled"] = True
                        return {
                            "success": True,
                            "message": f"{item['type'].capitalize()} '{item['label']}' (ID: {iid}) annulé"
                        }
                return {"success": False, "error": f"Aucun élément trouvé contenant '{search_term}'"}
            
            else:
                return {"success": False, "error": "Spécifiez un ID ou un terme de recherche"}
    
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
    
    def load_webpage(self, url: str, max_length: int = 8000) -> Dict[str, Any]:
        """
        Load and extract text content from a webpage.
        
        Args:
            url: URL of the webpage to load
            max_length: Maximum length of text to return (default 8000 chars)
            
        Returns:
            Dictionary with page content
        """
        print(f"[DEBUG Webpage] Loading: {url}")
        
        try:
            # Set a reasonable timeout and headers to appear as a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Try to get the encoding right
            if response.encoding is None or response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding or 'utf-8'
            
            html_content = response.text
            
            # Parse with BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return {
                    "success": False,
                    "error": "BeautifulSoup non installé. Installez avec: pip install beautifulsoup4"
                }
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Try to find main content area
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.post', '.entry']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.body if soup.body else soup
            
            # Extract text
            text = main_content.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[... contenu tronqué ...]"
            
            # Get page title
            title = soup.title.string if soup.title else ""
            
            print(f"[DEBUG Webpage] Loaded {len(text)} chars from {url}")
            
            return {
                "success": True,
                "url": url,
                "title": title.strip() if title else "",
                "content": text,
                "length": len(text)
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Timeout lors du chargement de {url}"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erreur lors du chargement: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur: {str(e)}"
            }
    
    def send_email(self, contact_name: str, subject: str, message: str) -> Dict[str, Any]:
        """
        Send an email via SMTP.
        
        Args:
            contact_name: Name of the contact (must be in EMAIL_CONTACTS)
            subject: Email subject
            message: Email body
            
        Returns:
            Dictionary with send status
        """
        print(f"[DEBUG Email] Tentative d'envoi à '{contact_name}': {subject}")
        
        if not self.email_smtp_server or not self.email_address or not self.email_password:
            error_msg = "Email non configuré. Ajoutez EMAIL_SMTP_SERVER, EMAIL_ADDRESS et EMAIL_PASSWORD dans config.py"
            print(f"[DEBUG Email] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        print(f"[DEBUG Email] Configuration OK")
        
        # Normalize contact name (lowercase, strip spaces)
        contact_name_normalized = contact_name.lower().strip()
        print(f"[DEBUG Email] Contact normalisé: '{contact_name_normalized}'")
        print(f"[DEBUG Email] Contacts disponibles: {list(self.email_contacts.keys())}")
        
        # Check if contact exists
        if contact_name_normalized not in self.email_contacts:
            available = ", ".join(self.email_contacts.keys())
            error_msg = f"Contact '{contact_name}' non trouvé. Contacts disponibles: {available}"
            print(f"[DEBUG Email] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        to_email = self.email_contacts[contact_name_normalized]
        print(f"[DEBUG Email] Adresse de destination: {to_email}")
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            print(f"[DEBUG Email] Création du message...")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain', 'utf-8'))
            
            print(f"[DEBUG Email] Connexion au serveur SMTP {self.email_smtp_server}:{self.email_smtp_port}...")
            
            # Connect and send
            with smtplib.SMTP(self.email_smtp_server, self.email_smtp_port) as server:
                server.starttls()
                print(f"[DEBUG Email] Authentification...")
                server.login(self.email_address, self.email_password)
                print(f"[DEBUG Email] Envoi...")
                server.send_message(msg)
            
            print(f"[DEBUG Email] Email envoyé avec succès!")
            
            return {
                "success": True,
                "contact": contact_name,
                "to_email": to_email,
                "subject": subject,
                "message": message
            }
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"Erreur d'authentification SMTP. Vérifiez votre mot de passe d'application: {str(e)}"
            print(f"[DEBUG Email] ERREUR: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Erreur lors de l'envoi: {str(e)}"
            print(f"[DEBUG Email] ERREUR: {error_msg}")
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
            
            # Return up to 10 articles, let GPT choose the most important ones
            articles = []
            for entry in feed.entries[:10]:
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:200] if entry.get("summary") else ""
                })
            
            print(f"[DEBUG News] Articles récupérés: {len(articles)}")
            if articles:
                print(f"[DEBUG News] Premier titre: {articles[0]['title'][:50]}...")
            
            return {
                "success": True,
                "category": category,
                "articles": articles,
                "total_results": len(articles),
                "instruction": "Sélectionne les 2-3 actualités les plus importantes et résume-les en environ 3 phrases au total. Ne liste pas tout."
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
            "name": "set_alarm",
            "description": "Définir une alarme/rappel pour une heure précise. Utiliser pour des rappels comme 'rappelle-moi à 18h30', 'réveille-moi à 7h demain', 'alarme à 14h00'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_str": {
                        "type": "string",
                        "description": "Heure de l'alarme au format HH:MM (24h). Ex: '18:30', '07:00', '14:00'"
                    },
                    "date_str": {
                        "type": "string",
                        "description": "Date optionnelle: 'aujourd'hui', 'demain', 'après-demain', ou format YYYY-MM-DD. Par défaut aujourd'hui (ou demain si l'heure est passée)"
                    },
                    "label": {
                        "type": "string",
                        "description": "Message de rappel optionnel (ex: 'aller chez le docteur', 'appeler maman', 'prendre les médicaments')"
                    }
                },
                "required": ["time_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Planifier une tâche à exécuter à une heure précise. La tâche sera exécutée automatiquement en arrière-plan. Utiliser pour des demandes comme 'envoie-moi la météo par mail à 7h', 'rappelle-moi d'aller au travail par SMS à 8h'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Description de la tâche à exécuter (ex: 'Envoie un email à moi avec la météo du jour', 'Envoie un SMS à moi pour rappeler d'aller au travail')"
                    },
                    "time_str": {
                        "type": "string",
                        "description": "Heure d'exécution au format HH:MM (24h). Ex: '07:00', '18:30'"
                    },
                    "date_str": {
                        "type": "string",
                        "description": "Date optionnelle: 'aujourd'hui', 'demain', 'après-demain', ou format YYYY-MM-DD"
                    },
                    "speak_result": {
                        "type": "boolean",
                        "description": "Si true, le résultat sera prononcé à voix haute. Si false (défaut), la tâche s'exécute silencieusement."
                    }
                },
                "required": ["task", "time_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task_delay",
            "description": "Planifier une tâche à exécuter après un délai. Utiliser pour des demandes comme 'envoie-moi la météo par SMS dans 30 secondes', 'dans 5 minutes dis-moi les news'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Description de la tâche à exécuter (ex: 'Envoie un SMS à moi avec la météo', 'Dis-moi les actualités')"
                    },
                    "seconds": {
                        "type": "integer",
                        "description": "Nombre de secondes avant d'exécuter la tâche (ex: pour '2 minutes', utiliser 120)"
                    },
                    "speak_result": {
                        "type": "boolean",
                        "description": "Si true, le résultat sera prononcé à voix haute. Si false (défaut), la tâche s'exécute silencieusement."
                    }
                },
                "required": ["task", "seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_scheduled",
            "description": "Lister tous les éléments programmés en attente: timers, alarmes et tâches planifiées.",
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
            "name": "cancel_scheduled",
            "description": "Annuler un timer, une alarme ou une tâche planifiée. Peut annuler par ID ou par recherche dans le label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "integer",
                        "description": "ID de l'élément à annuler"
                    },
                    "search_term": {
                        "type": "string",
                        "description": "Terme de recherche pour trouver l'élément par son label (annule le premier résultat)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Rechercher des informations sur le web. Retourne une liste de résultats avec titre, lien et extrait. Pour obtenir plus de détails sur un résultat, utiliser load_webpage avec l'URL.",
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
            "name": "load_webpage",
            "description": "Charger et lire le contenu d'une page web. Utiliser après search_web pour obtenir plus de détails sur un résultat, ou directement si l'URL est connue. Utile pour trouver des informations spécifiques comme numéros de téléphone, horaires, avis, disponibilités, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL complète de la page à charger (ex: 'https://example.com/page')"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Envoyer un email à un contact. Le nom du contact doit correspondre exactement à un contact configuré.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "Nom du contact (ex: 'moi', 'marie', 'papa', 'maman'). Doit être en minuscules."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Sujet de l'email"
                    },
                    "message": {
                        "type": "string",
                        "description": "Le contenu de l'email"
                    }
                },
                "required": ["contact_name", "subject", "message"]
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
            "description": "Obtenir les dernières actualités françaises. Retourne plusieurs articles parmi lesquels tu dois choisir les 2-3 plus importants et les résumer brièvement.",
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
