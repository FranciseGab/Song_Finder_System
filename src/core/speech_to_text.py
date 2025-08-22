import pyaudio
import numpy as np
import time
import speech_recognition as sr
import requests
import json
import tempfile
import wave
import os

class SpeechRecognizer:
    def __init__(self, use_whisper=False, whisper_api_key=None):
        """ 
        Initialize speech recognizer
        use_whisper: Whether to use Whisper API (requires API key)
        whisper_api_key: OpenAI API key for Whisper (if using API)
        """
        self.use_whisper_api = use_whisper
        self.whisper_api_key = whisper_api_key
        
        # Initialize speech_recognition as default
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.timeout = 5
        self.phrase_timeout = 3
        
        # Adjust for ambient noise
        self.calibrate_microphone()
    
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                print("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Microphone calibrated.")
        except Exception as e:
            print(f"Microphone calibration warning: {e}")
    
    def record_audio(self, record_seconds=5):
        """Record audio using PyAudio as fallback"""
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            
            print("Recording... (speak now)")
            frames = []
            
            for _ in range(0, int(16000 / 1024 * record_seconds)):
                data = stream.read(1024)
                frames.append(data)
            
            print("Recording finished")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                with wave.open(tmp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(b''.join(frames))
                return tmp_file.name
                
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
    
    def transcribe_with_whisper_api(self, audio_file_path):
        """Transcribe audio using Whisper API"""
        if not self.whisper_api_key:
            print("Whisper API key not provided")
            return None
            
        try:
            with open(audio_file_path, 'rb') as audio_file:
                response = requests.post(
                    'https://api.openai.com/v1/audio/transcriptions',
                    headers={
                        'Authorization': f'Bearer {self.whisper_api_key}'
                    },
                    files={
                        'file': audio_file
                    },
                    data={
                        'model': 'whisper-1',
                        'language': 'en'
                    }
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '').strip()
            else:
                print(f"Whisper API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling Whisper API: {e}")
            return None
    
    def listen_and_convert(self):
        """
        Listen to microphone input and convert to text
        Returns: Converted text or None if failed
        """
        try:
            print("Listening for speech...")
            
            with self.microphone as source:
                # Listen for audio input
                audio = self.recognizer.listen(
                    source, 
                    timeout=self.timeout, 
                    phrase_time_limit=self.phrase_timeout
                )
            
            print("Processing audio...")
            
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")
                return text.lower().strip()
            
            except sr.UnknownValueError:
                print("Google could not understand the audio")
                # Fallback to other services
                pass
            
            except sr.RequestError as e:
                print(f"Error with Google service: {e}")
                # Fallback to other services
                pass
            
            # Try Sphinx (offline) as fallback
            try:
                text = self.recognizer.recognize_sphinx(audio)
                print(f"Offline recognition: {text}")
                return text.lower().strip()
            except:
                print("Offline recognition also failed")
                return None
        
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return None
        
        except Exception as e:
            print(f"Speech recognition error: {e}")
            
            # Final fallback: record audio and try Whisper API if enabled
            if self.use_whisper_api:
                print("Trying Whisper API as fallback...")
                audio_file = self.record_audio()
                if audio_file:
                    text = self.transcribe_with_whisper_api(audio_file)
                    # Clean up temporary file
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                    return text
            return None
    
    def listen_for_search(self):
        """Capture voice input and convert to text"""
        return self.listen_and_convert()
    
    def listen_continuous(self, callback, stop_listening=None):
        """
        Continuous listening mode
        Args:
            callback: Function to call with recognized text
            stop_listening: Function that returns True to stop listening
        """
        print("Starting continuous listening mode...")
        
        while True:
            if stop_listening and stop_listening():
                break
            
            text = self.listen_and_convert()
            if text:
                callback(text)
            
            time.sleep(0.1)
    
    def test_microphone(self):
        """Test microphone functionality"""
        try:
            with self.microphone as source:
                print("Testing microphone...")
                audio = self.recognizer.listen(source, timeout=2)
                print("Microphone test successful")
                return True
        except Exception as e:
            print(f"Microphone test failed: {e}")
            return False
    
    def get_available_microphones(self):
        """Get list of available microphones"""
        mic_list = []
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            mic_list.append(f"{index}: {name}")
        return mic_list
    
    def set_microphone_index(self, index):
        """Set specific microphone by index"""
        try:
            self.microphone = sr.Microphone(device_index=index)
            self.calibrate_microphone()
            return True
        except Exception as e:
            print(f"Failed to set microphone {index}: {e}")
            return False