from typing import Dict, Any

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown"
}

class OutputPostProcessor:
    """
    Validates, normalizes, and sanitizes router outputs to guarantee 100% compliance
    with HackerRank submission specifications.
    """
    def sanitize_reason(self, reason: str) -> str:
        if not reason:
            return "Standard notification decision based on user activity context."
        
        # Clean double spaces, trailing dots
        clean = reason.strip().replace("\n", " ").replace("  ", " ")
        if not clean.endswith("."):
            clean += "."
        return clean

    def process(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        action = prediction.get("action", "digest").lower()
        if action not in ALLOWED_ACTIONS:
            action = "digest"

        msg_type = prediction.get("message_type", "unknown").lower()
        if msg_type not in ALLOWED_TYPES:
            msg_type = "unknown"

        reason = self.sanitize_reason(prediction.get("reason", ""))

        try:
            confidence = float(prediction.get("confidence", 0.80))
            confidence = round(max(0.0, min(1.0, confidence)), 2)
        except (ValueError, TypeError):
            confidence = 0.80

        evidence = str(prediction.get("evidence_message_ids", "none")).strip()
        if not evidence:
            evidence = "none"

        return {
            "message_id": prediction["message_id"],
            "action": action,
            "message_type": msg_type,
            "reason": reason,
            "confidence": confidence,
            "evidence_message_ids": evidence
        }
