import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

class HybridMediaProcessor:
    """
    Hybrid Multimodal Media Processor for Images and Voice Notes.
    Features persistent disk caching (.cache/media_cache.json) with MD5 hash lookup.
    """
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "media_cache.json"
        self._load_env_file()
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_env_file(self):
        # Auto-load .env file from project root if present
        env_path = self.cache_dir.parent / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip().strip("'").strip('"')
            except Exception:
                pass

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    def _get_file_hash(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return hashlib.md5(file_path.encode("utf-8")).hexdigest()
        
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def process_image(self, image_path: Optional[str]) -> Dict[str, Any]:
        """
        Process an image message using VLM / OCR with caching.
        Returns:
            {
                "extracted_text": str,
                "detected_intent": str,
                "is_scam_suspect": bool,
                "has_urgent_signals": bool,
                "vlm_used": bool
            }
        """
        if not image_path or not os.path.exists(image_path):
            return {
                "extracted_text": "",
                "detected_intent": "unknown",
                "is_scam_suspect": False,
                "has_urgent_signals": False,
                "vlm_used": False
            }

        file_hash = self._get_file_hash(image_path)

        # Cache Hit Check
        if file_hash in self.cache:
            return self.cache[file_hash]

        # Cache Miss: Attempt VLM API / Local OCR
        result = self._extract_image_features(image_path)
        
        # Store in cache
        self.cache[file_hash] = result
        self._save_cache()
        return result

    def _extract_image_features(self, image_path: str) -> Dict[str, Any]:
        # Check for Gemini / OpenAI API key for VLM
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if gemini_key:
            try:
                # Attempt VLM via Gemini dynamically
                import importlib
                genai = importlib.import_module("google.generativeai")
                pil_module = importlib.import_module("PIL")
                Image = pil_module.Image

                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                img = Image.open(image_path)
                prompt = (
                    "Analyze this WhatsApp image. Return JSON with keys:\n"
                    "extracted_text: all visible text in image\n"
                    "detected_intent: 'promotion', 'receipt', 'event', 'scam', 'greeting', or 'personal'\n"
                    "is_scam_suspect: boolean\n"
                    "has_urgent_signals: boolean"
                )
                response = model.generate_content([prompt, img])
                text_resp = response.text.strip()
                # Parse JSON if possible
                if "{" in text_resp and "}" in text_resp:
                    json_str = text_resp[text_resp.find("{"):text_resp.rfind("}")+1]
                    parsed = json.loads(json_str)
                    parsed["vlm_used"] = True
                    return parsed
            except Exception:
                pass

        # Fallback to local OCR / PIL analysis
        try:
            import importlib
            pytesseract = importlib.import_module("pytesseract")
            pil_module = importlib.import_module("PIL")
            Image = pil_module.Image

            img = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(img).strip()
            
            lower_text = ocr_text.lower()
            is_scam = any(w in lower_text for w in ["win lottery", "claim prize", "urgent bank block", "verify otp", "wire money"])
            is_urgent = any(w in lower_text for w in ["urgent", "due today", "immediately", "deadline", "emergency"])
            
            intent = "promotion" if any(w in lower_text for w in ["off", "discount", "sale", "buy 1 get 1"]) else "event"
            
            return {
                "extracted_text": ocr_text,
                "detected_intent": intent if ocr_text else "unknown",
                "is_scam_suspect": is_scam,
                "has_urgent_signals": is_urgent,
                "vlm_used": False
            }
        except Exception:
            # Baseline Heuristic for offline image processing without OCR binary
            filename = os.path.basename(image_path).lower()
            return {
                "extracted_text": f"Image attachment {filename}",
                "detected_intent": "unknown",
                "is_scam_suspect": False,
                "has_urgent_signals": False,
                "vlm_used": False
            }

    def process_voice_note(self, audio_path: Optional[str]) -> Dict[str, Any]:
        """
        Process a voice note message using STT / Audio Metadata profiling.
        Returns:
            {
                "transcription": str,
                "detected_intent": str,
                "has_urgent_signals": bool,
                "duration_sec": float,
                "stt_used": bool
            }
        """
        if not audio_path or not os.path.exists(audio_path):
            return {
                "transcription": "",
                "detected_intent": "unknown",
                "has_urgent_signals": False,
                "duration_sec": 0.0,
                "stt_used": False
            }

        file_hash = self._get_file_hash(audio_path)

        if file_hash in self.cache:
            return self.cache[file_hash]

        result = self._extract_audio_features(audio_path)
        self.cache[file_hash] = result
        self._save_cache()
        return result

    def _extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        # Estimate duration from file size if audio libraries aren't present
        file_size = os.path.getsize(audio_path)
        est_duration = round(file_size / 4000.0, 1)  # Rough estimate for MP3/AAC

        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                import importlib
                openai = importlib.import_module("openai")

                client = openai.OpenAI(api_key=openai_key)
                with open(audio_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                text = transcript.text
                lower_text = text.lower()
                is_urgent = any(w in lower_text for w in ["call me", "urgent", "emergency", "immediately", "where are you"])
                return {
                    "transcription": text,
                    "detected_intent": "urgent" if is_urgent else "personal",
                    "has_urgent_signals": is_urgent,
                    "duration_sec": est_duration,
                    "stt_used": True
                }
            except Exception:
                pass

        # Offline / Fallback
        return {
            "transcription": f"Voice note audio ({est_duration}s)",
            "detected_intent": "personal",
            "has_urgent_signals": False,
            "duration_sec": est_duration,
            "stt_used": False
        }
