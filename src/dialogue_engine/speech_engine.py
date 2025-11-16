"""
Speech Recognition and Text-to-Speech integration
Supports voice interaction for the dialogue system
"""
import speech_recognition as sr
import pyttsx3
from typing import Optional
from utils.config_loader import config
from utils.logger import setup_logger

logger = setup_logger()


class SpeechEngine:
    """
    Handles speech recognition (ASR) and text-to-speech (TTS)
    """
    
    def __init__(self):
        """Initialize speech engine"""
        self.asr_language = config.get("speech.asr.language", "zh-CN")
        self.tts_language = config.get("speech.tts.language", "zh-CN")
        self.tts_rate = config.get("speech.tts.rate", 150)
        self.tts_volume = config.get("speech.tts.volume", 0.9)
        
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.tts_rate)
            self.tts_engine.setProperty('volume', self.tts_volume)
            self._tts_available = True
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize TTS engine: {e}")
            self._tts_available = False
    
    def speech_to_text(
        self,
        audio_source: Optional[str] = None,
        use_microphone: bool = True
    ) -> Optional[str]:
        """
        Convert speech to text
        
        Args:
            audio_source: Path to audio file (if not using microphone)
            use_microphone: Whether to use microphone for input
            
        Returns:
            Recognized text or None if recognition fails
        """
        try:
            if use_microphone:
                # Use microphone
                with sr.Microphone() as source:
                    logger.info("Listening for speech...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            else:
                # Use audio file
                if not audio_source:
                    logger.error("Audio source path required when not using microphone")
                    return None
                
                with sr.AudioFile(audio_source) as source:
                    audio = self.recognizer.record(source)
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio, language=self.asr_language)
            logger.info(f"Recognized text: {text}")
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out - no speech detected")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return None
    
    def text_to_speech(self, text: str, output_file: Optional[str] = None) -> bool:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            output_file: Optional path to save audio file
            
        Returns:
            True if successful, False otherwise
        """
        if not self._tts_available:
            logger.warning("TTS engine not available")
            return False
        
        try:
            if output_file:
                # Save to file
                self.tts_engine.save_to_file(text, output_file)
                self.tts_engine.runAndWait()
                logger.info(f"Speech saved to {output_file}")
            else:
                # Play directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                logger.info("Speech played successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return False
    
    def is_available(self) -> dict:
        """Check availability of speech features"""
        return {
            "asr": True,  # ASR via speech_recognition library
            "tts": self._tts_available
        }


class VoiceInteractionHandler:
    """
    High-level handler for voice-based dialogue interaction
    """
    
    def __init__(self, dialogue_engine):
        """
        Initialize voice interaction handler
        
        Args:
            dialogue_engine: Instance of FinancialDialogueEngine
        """
        self.dialogue_engine = dialogue_engine
        self.speech_engine = SpeechEngine()
        logger.info("VoiceInteractionHandler initialized")
    
    def voice_query(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        use_microphone: bool = True,
        audio_file: Optional[str] = None,
        speak_response: bool = True
    ) -> dict:
        """
        Process a voice query end-to-end
        
        Args:
            user_id: User identifier
            session_id: Optional session ID
            use_microphone: Whether to use microphone input
            audio_file: Path to audio file (if not using microphone)
            speak_response: Whether to speak the response
            
        Returns:
            Response dictionary
        """
        # Convert speech to text
        query_text = self.speech_engine.speech_to_text(
            audio_source=audio_file,
            use_microphone=use_microphone
        )
        
        if not query_text:
            return {
                "success": False,
                "error": "Failed to recognize speech",
                "response": "抱歉，我没有听清楚您的问题，请再说一遍。"
            }
        
        # Process query through dialogue engine
        result = self.dialogue_engine.process_query(
            user_id=user_id,
            query=query_text,
            session_id=session_id
        )
        
        # Speak response if requested
        if speak_response:
            self.speech_engine.text_to_speech(result["response"])
        
        return {
            "success": True,
            "query": query_text,
            **result
        }
