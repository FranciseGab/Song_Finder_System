"""
Speech-to-Text Module
Handles audio input and conversion to text
"""

import speech_recognition as sr
import pyaudio
import time

class SpeechRecognizer:
    """Handles speech recognition functionality"""
    
    def __init__(self, timeout=5, phrase_timeout=2):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.timeout = timeout
        self.phrase_timeout = phrase_timeout
        
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
            
            # Convert speech to text using Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")
                return text.lower().strip()
            
            except sr.UnknownValueError:
                print("Could not understand the audio")
                return None
            
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                # Fallback to offline recognition if available
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    print(f"Offline recognition: {text}")
                    return text.lower().strip()
                except:
                    return None
        
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return None
        
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None
    
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
            
            time.sleep(0.1)  # Brief pause between attempts
    
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

# Simple audio recorder class for basic functionality
class SimpleAudioRecorder:
    """Fallback audio recorder if speech_recognition fails"""
    
    def __init__(self, sample_rate=44100, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
    
    def record_audio(self, duration=5):
        """Record audio for specified duration"""
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"Recording for {duration} seconds...")
            frames = []
            
            for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            print("Recording complete")
            return b''.join(frames)
        
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    def __del__(self):
        """Cleanup audio resources"""
        try:
            self.audio.terminate()
        except:
            pass