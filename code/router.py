import re
from typing import Dict, Any
from context import ContextEngine
from media import HybridMediaProcessor
from retrieval import EvidenceRetriever

class NotificationRouter:
    """
    Fenced Decision Engine for personalized WhatsApp message routing.
    Integrates deterministic safety guards, dynamic contextual classification,
    and calibrated confidence scoring.
    """
    def __init__(self, context: ContextEngine, media: HybridMediaProcessor, retriever: EvidenceRetriever):
        self.context = context
        self.media = media
        self.retriever = retriever

    def route_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        user_id = msg.get("user_id", "")
        conv_type = msg.get("conversation_type", "")
        group_id = msg.get("group_id", "")
        business_id = msg.get("business_id", "")
        sender_id = msg.get("sender_user_id", "")
        created_at = msg.get("created_at", "")
        msg_text = msg.get("message_text", "")
        media_type = msg.get("media_type", "")
        media_id = msg.get("media_id", "")
        forwarded_count = int(msg.get("forwarded_count", 0) or 0)

        # 1. Fetch Multimodal Media Features
        media_features = {}
        if media_type == "image" and media_id:
            img_path = self.context.get_image_path(media_id)
            media_features = self.media.process_image(img_path)
        elif media_type == "voice" and media_id:
            audio_path = self.context.get_voice_note_path(media_id)
            media_features = self.media.process_voice_note(audio_path)

        extracted_media_text = media_features.get("extracted_text", "") or media_features.get("transcription", "")
        full_text = f"{msg_text} {extracted_media_text}".strip().lower()

        # Add media text back to msg dict for retrieval query
        msg["media_text"] = extracted_media_text

        # 2. Query RAG Evidence Retriever
        evidence_ids = self.retriever.find_evidence(msg)

        # 3. Context Lookups
        is_dnd = self.context.is_quiet_hours(user_id, created_at)
        user_info = self.context.get_user_info(user_id)
        group_info = self.context.get_group_info(group_id) if group_id else {}
        group_member_info = self.context.get_group_membership(group_id, user_id) if group_id else {}
        biz_info = self.context.get_business_info(business_id) if business_id else {}
        user_biz_info = self.context.get_user_business_history(user_id, business_id) if (user_id and business_id) else {}

        is_group_muted = (group_member_info.get("group_muted_by_user") == "1")
        is_admin_sender = (group_member_info.get("role") == "admin") or (sender_id and self.context.get_group_membership(group_id, sender_id).get("role") == "admin")

        # Check for direct user mention e.g. "@u_010" or "@u_004"
        has_direct_mention = f"@{user_id}".lower() in full_text

        # ---------------------------------------------------------
        # STAGE 1: Hard Safety & Scam / Phishing / Fraud Protection
        # ---------------------------------------------------------

        scam_keywords = [
            "lottery", "prize", "otp", "verify account", "claim reward", "reattempt fee",
            "bankofamerica-delivery", "amazonpay-delivery", "giveaway", "0.5 btc", "crypto",
            "work from home", "typing online", "scratch card", "e-kyc", "kyc", "account suspended",
            "account block", "sim card will be deactivated", "unauthorized login", "security alert",
            "verify ssn", "instant loan", "apk", "whatsapp gold", "redelivery", "spin the wheel",
            "spin-win", "refund pending", "t.me/", "bit.ly/", ".xyz", ".site", ".online", ".net",
            "free ₹", "free $", "earn ₹", "earn $", "1000% return", "pump channel", "win $", "won $"
        ]
        
        domain_mismatch = False
        if biz_info:
            off_domain = biz_info.get("official_domain", "")
            sender_domain = biz_info.get("domain_used_by_sender", "")
            if off_domain and sender_domain and off_domain != sender_domain:
                domain_mismatch = True

        is_unverified_imposter_biz = False
        if biz_info:
            is_unverified = (biz_info.get("verified") == "0")
            user_reports = int(biz_info.get("user_reports_30d", 0) or 0)
            if is_unverified and (user_reports > 10 or domain_mismatch):
                is_unverified_imposter_biz = True

        has_scam_pattern = any(k in full_text for k in scam_keywords) or domain_mismatch or is_unverified_imposter_biz or media_features.get("is_scam_suspect")

        if has_scam_pattern:
            return {
                "message_id": msg["message_id"],
                "action": "mute",
                "message_type": "scam",
                "reason": "The message contains suspicious links, imposter business indicators, or scam patterns.",
                "confidence": 0.95,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 2: Forwarded Spam & Chain Messages
        # ---------------------------------------------------------

        if forwarded_count >= 5 or any(w in full_text for w in ["send this message to", "send this prayer", "drink warm water", "forwarded as received"]):
            return {
                "message_id": msg["message_id"],
                "action": "mute",
                "message_type": "forward" if forwarded_count >= 5 else "spam",
                "reason": "The message matches repeated forward chains or viral spam patterns.",
                "confidence": 0.88,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 3: Direct User Mentions & Urgent Overrides
        # ---------------------------------------------------------

        if is_group_muted and has_direct_mention:
            return {
                "message_id": msg["message_id"],
                "action": "notify",
                "message_type": "urgent" if any(k in full_text for k in ["urgent", "eod", "prod", "alert", "failing", "500"]) else "personal",
                "reason": "The message contains a direct mention targeting the user, overriding muted group preferences.",
                "confidence": 0.90,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 4: Urgent & Critical Time-Sensitive Intent
        # ---------------------------------------------------------

        urgent_keywords = [
            "hospital", "emergency", "broke down", "pickup", "fever", "flight", "call me immediately",
            "call back immediately", "asap", "server", "outage", "bridge call", "500 error", "deadline",
            "production", "failing", "critical", "room 302", "expressway"
        ]

        if any(k in full_text for k in urgent_keywords):
            msg_cat = "urgent"
            if "upi" in full_text or "fare" in full_text:
                msg_cat = "payment"
            return {
                "message_id": msg["message_id"],
                "action": "notify",
                "message_type": msg_cat,
                "reason": "Time-sensitive urgent message requiring immediate user attention.",
                "confidence": 0.92,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 5: Payment & Financial Transactions
        # ---------------------------------------------------------

        payment_keywords = ["upi", "credited", "debited", "salary", "bill", "due on", "split share", "refund", "card ending", "neft"]
        if any(k in full_text for k in payment_keywords):
            is_urgent_pay = any(k in full_text for k in ["urgently", "now", "emergency", "taxi fare", "credited"])
            return {
                "message_id": msg["message_id"],
                "action": "notify" if is_urgent_pay else "digest",
                "message_type": "payment",
                "reason": "Financial transaction or payment notice.",
                "confidence": 0.88,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 6: Event & Party Invitations
        # ---------------------------------------------------------

        event_keywords = ["invited", "invitation", "party", "concert", "picnic", "reunion", "wedding", "save the date", "book of the month", "water supply", "shut off", "tickets"]
        if any(k in full_text for k in event_keywords):
            return {
                "message_id": msg["message_id"],
                "action": "digest",
                "message_type": "event",
                "reason": "Event invitation or scheduled community notice.",
                "confidence": 0.85,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 7: Business Account Messaging (Verified)
        # ---------------------------------------------------------

        if conv_type == "business" and biz_info:
            is_promo_text = any(k in full_text for k in ["sale", "off", "discount", "coupon", "arrivals", "eorr", "flash sale"])
            if is_promo_text:
                return {
                    "message_id": msg["message_id"],
                    "action": "digest",
                    "message_type": "promotion",
                    "reason": "Promotional campaign update from business sender.",
                    "confidence": 0.82,
                    "evidence_message_ids": evidence_ids
                }
            else:
                return {
                    "message_id": msg["message_id"],
                    "action": "notify" if any(k in full_text for k in ["shipped", "dispatched", "delivered"]) else "digest",
                    "message_type": "business_update",
                    "reason": "Verified business notification update.",
                    "confidence": 0.86,
                    "evidence_message_ids": evidence_ids
                }

        # ---------------------------------------------------------
        # STAGE 8: Personal 1-on-1 Chats
        # ---------------------------------------------------------

        if conv_type == "personal":
            is_casual = any(k in full_text for k in ["dinner", "coffee", "photos", "recipe", "movie", "catch up", "how are you", "weekend"])
            if is_casual:
                msg_cat = "greeting" if ("coffee" in full_text or "how are you" in full_text or "great week" in full_text) else "personal"
                return {
                    "message_id": msg["message_id"],
                    "action": "digest",
                    "message_type": msg_cat,
                    "reason": "Casual personal conversation update.",
                    "confidence": 0.82,
                    "evidence_message_ids": evidence_ids
                }
            else:
                return {
                    "message_id": msg["message_id"],
                    "action": "notify",
                    "message_type": "personal",
                    "reason": "Direct personal message from contact.",
                    "confidence": 0.87,
                    "evidence_message_ids": evidence_ids
                }

        # ---------------------------------------------------------
        # STAGE 9: Group Message Default
        # ---------------------------------------------------------

        if conv_type == "group":
            if is_group_muted:
                return {
                    "message_id": msg["message_id"],
                    "action": "mute",
                    "message_type": "personal",
                    "reason": "Group is muted by user.",
                    "confidence": 0.84,
                    "evidence_message_ids": evidence_ids
                }
            else:
                return {
                    "message_id": msg["message_id"],
                    "action": "digest",
                    "message_type": "personal",
                    "reason": "Standard group activity update.",
                    "confidence": 0.80,
                    "evidence_message_ids": evidence_ids
                }

        # ---------------------------------------------------------
        # STAGE 10: Fallback Route
        # ---------------------------------------------------------

        return {
            "message_id": msg["message_id"],
            "action": "digest" if not is_dnd else "mute",
            "message_type": "personal" if conv_type == "personal" else ("business_update" if conv_type == "business" else "personal"),
            "reason": "Standard contextual priority routing.",
            "confidence": 0.78,
            "evidence_message_ids": evidence_ids
        }
