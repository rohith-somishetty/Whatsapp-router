import os
import json
import time
import hashlib
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class HybridMediaProcessor:
    """
    Hybrid Multimodal Media Processor for Images and Voice Notes.
    Features exponential backoff retries, local FFmpeg PATH auto-injection,
    6.0s rate-limiting delay, and persistent disk caching (.cache/media_cache.json).
    """
    def __init__(self, cache_dir: str = ".cache", max_rpm: int = 10, max_daily_calls: int = 1400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "media_cache.json"
        
        self.delay_between_calls = 60.0 / max_rpm if max_rpm > 0 else 6.0
        self.max_daily_calls = max_daily_calls
        self.last_api_call_time = 0.0

        self._load_env_file()
        self._ensure_local_ffmpeg_path()
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _ensure_local_ffmpeg_path(self):
        """Auto-inject imageio-ffmpeg binary path into os.environ['PATH'] if system ffmpeg is missing."""
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_bin)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.path.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

        # Also check standard Tesseract installation paths
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR",
            r"C:\Program Files (x86)\Tesseract-OCR",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR")
        ]
        for t_dir in tesseract_paths:
            t_exe = os.path.join(t_dir, "tesseract.exe")
            if os.path.exists(t_exe) and t_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = t_dir + os.path.pathsep + os.environ.get("PATH", "")

    def _load_env_file(self):
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

    def _check_daily_quota(self) -> bool:
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_stats = self.cache.get("__quota_stats__", {})
        
        if daily_stats.get("date") != today_str:
            daily_stats = {"date": today_str, "calls": 0}
            
        if daily_stats.get("calls", 0) >= self.max_daily_calls:
            return False
            
        daily_stats["calls"] = daily_stats.get("calls", 0) + 1
        self.cache["__quota_stats__"] = daily_stats
        self._save_cache()
        return True

    def _enforce_rate_limit(self):
        elapsed = time.time() - self.last_api_call_time
        if elapsed < self.delay_between_calls:
            time.sleep(self.delay_between_calls - elapsed)
        self.last_api_call_time = time.time()

    def _get_file_hash(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return hashlib.md5(file_path.encode("utf-8")).hexdigest()
        
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def process_image(self, image_path: Optional[str]) -> Dict[str, Any]:
        if not image_path or not os.path.exists(image_path):
            return {
                "extracted_text": "",
                "detected_intent": "unknown",
                "is_scam_suspect": False,
                "has_urgent_signals": False,
                "vlm_used": False,
                "fallback_used": True
            }

        file_hash = self._get_file_hash(image_path)
        if file_hash in self.cache:
            return self.cache[file_hash]

        result = self._extract_image_features(image_path)
        self.cache[file_hash] = result
        self._save_cache()
        return result

    def _extract_image_features(self, image_path: str) -> Dict[str, Any]:
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if gemini_key and self._check_daily_quota():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    import importlib
                    genai_pkg = importlib.import_module("google.genai")
                    pil_module = importlib.import_module("PIL")
                    Image = pil_module.Image

                    self._enforce_rate_limit()

                    client = genai_pkg.Client(api_key=gemini_key)
                    img = Image.open(image_path)
                    prompt = (
                        "Analyze this WhatsApp image. Return JSON with keys:\n"
                        "extracted_text: all visible text in image\n"
                        "detected_intent: 'promotion', 'receipt', 'event', 'scam', 'greeting', or 'personal'\n"
                        "is_scam_suspect: boolean\n"
                        "has_urgent_signals: boolean"
                    )
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[prompt, img]
                    )
                    text_resp = response.text.strip()
                    if "{" in text_resp and "}" in text_resp:
                        json_str = text_resp[text_resp.find("{"):text_resp.rfind("}")+1]
                        parsed = json.loads(json_str)
                        parsed["vlm_used"] = True
                        parsed["fallback_used"] = False
                        return parsed
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        backoff_sleep = 4.0 * (2 ** attempt)
                        time.sleep(backoff_sleep)
                    else:
                        break

        # Offline local OCR / PIL fallback
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
                "vlm_used": False,
                "fallback_used": False
            }
        except Exception:
            filename = os.path.basename(image_path).lower()
            return {
                "extracted_text": f"Image attachment {filename}",
                "detected_intent": "unknown",
                "is_scam_suspect": False,
                "has_urgent_signals": False,
                "vlm_used": False,
                "fallback_used": True
            }

    def process_voice_note(self, audio_path: Optional[str]) -> Dict[str, Any]:
        if not audio_path or not os.path.exists(audio_path):
            return {
                "transcription": "",
                "detected_intent": "unknown",
                "has_urgent_signals": False,
                "duration_sec": 0.0,
                "stt_used": False,
                "fallback_used": True
            }

        file_hash = self._get_file_hash(audio_path)
        if file_hash in self.cache:
            return self.cache[file_hash]

        result = self._extract_audio_features(audio_path)
        self.cache[file_hash] = result
        self._save_cache()
        return result

    def _extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        file_size = os.path.getsize(audio_path)
        est_duration = round(file_size / 4000.0, 1)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("WHISPER_API_KEY")
        if hf_token and self._check_daily_quota():
            try:
                self._enforce_rate_limit()
                url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
                with open(audio_path, "rb") as f:
                    data = f.read()
                req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {hf_token}", "Content-Type": "audio/mpeg"})
                with urllib.request.urlopen(req) as resp:
                    res = json.loads(resp.read().decode('utf-8'))
                    text = res.get("text", "").strip()
                    if text:
                        lower_text = text.lower()
                        is_urgent = any(w in lower_text for w in ["call me", "urgent", "emergency", "immediately", "where are you", "hospital", "accident", "help"])
                        return {
                            "transcription": text,
                            "detected_intent": "urgent" if is_urgent else "personal",
                            "has_urgent_signals": is_urgent,
                            "duration_sec": est_duration,
                            "stt_used": True,
                            "model_used": "hf_whisper_large_v3",
                            "fallback_used": False
                        }
            except Exception:
                pass

        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key and self._check_daily_quota():
            try:
                self._enforce_rate_limit()
                import importlib
                groq_pkg = importlib.import_module("groq")
                client = groq_pkg.Groq(api_key=groq_key)
                with open(audio_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(audio_path), audio_file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                    )
                text = transcription.text.strip()
                lower_text = text.lower()
                is_urgent = any(w in lower_text for w in ["call me", "urgent", "emergency", "immediately", "where are you", "hospital", "accident", "help"])
                return {
                    "transcription": text,
                    "detected_intent": "urgent" if is_urgent else "personal",
                    "has_urgent_signals": is_urgent,
                    "duration_sec": est_duration,
                    "stt_used": True,
                    "model_used": "groq_whisper_large_v3",
                    "fallback_used": False
                }
            except Exception:
                pass

        try:
            import importlib
            transformers = importlib.import_module("transformers")
            pipeline = transformers.pipeline

            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                chunk_length_s=30
            )
            res = asr_pipeline(audio_path)
            text = res.get("text", "").strip()
            lower_text = text.lower()
            is_urgent = any(w in lower_text for w in ["call me", "urgent", "emergency", "immediately", "where are you", "hospital", "accident", "help"])

            return {
                "transcription": text,
                "detected_intent": "urgent" if is_urgent else "personal",
                "has_urgent_signals": is_urgent,
                "duration_sec": est_duration,
                "stt_used": True,
                "model_used": "local_whisper_tiny",
                "fallback_used": False
            }
        except Exception:
            pass

        return {
            "transcription": f"Voice note audio ({est_duration}s)",
            "detected_intent": "personal",
            "has_urgent_signals": False,
            "duration_sec": est_duration,
            "stt_used": False,
            "model_used": "metadata_fallback",
            "fallback_used": True
        }
