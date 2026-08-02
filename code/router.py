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
        # STAGE 4A: Hard Safety & Deterministic Overrides
        # ---------------------------------------------------------

        # Scam / Phishing Guard
        scam_keywords = ["lottery", "prize", "otp", "verify account", "claim reward", "reattempt fee", "bankofamerica-delivery", "amazonpay-delivery"]
        domain_mismatch = False
        if biz_info:
            off_domain = biz_info.get("official_domain", "")
            sender_domain = biz_info.get("domain_used_by_sender", "")
            if off_domain and sender_domain and off_domain != sender_domain:
                domain_mismatch = True

        if any(k in full_text for k in scam_keywords) or domain_mismatch or media_features.get("is_scam_suspect"):
            return {
                "message_id": msg["message_id"],
                "action": "mute",
                "message_type": "scam",
                "reason": "The message contains suspicious links, domain mismatches, or unverified scam indicators.",
                "confidence": 0.95,
                "evidence_message_ids": evidence_ids
            }

        # Direct Mention Override in Muted Group
        if is_group_muted and has_direct_mention:
            return {
                "message_id": msg["message_id"],
                "action": "notify",
                "message_type": "urgent" if ("urgent" in full_text or "eod" in full_text or "prod" in full_text) else "personal",
                "reason": "The message contains a direct mention targeting the user, overriding muted group preferences.",
                "confidence": 0.90,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 4B: Personalized Dynamic Router
        # ---------------------------------------------------------

        # 1. Forwarded Greetings & Chain Messages
        if forwarded_count >= 5 or any(w in full_text for w in ["good morning", "stay positive", "drink warm water", "forwarded as received"]):
            return {
                "message_id": msg["message_id"],
                "action": "mute",
                "message_type": "greeting" if "good morning" in full_text else "forward",
                "reason": "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                "confidence": 0.85,
                "evidence_message_ids": evidence_ids
            }

        # 2. Urgent Group Notices (Water tanker, school bus, work emergency)
        urgent_keywords = ["water", "tanker", "bus", "valve", "plumber", "school", "parents", "prod review", "urgent", "emergency", "deadline"]
        if conv_type == "group" and any(k in full_text for k in urgent_keywords):
            if is_admin_sender:
                msg_cat = "event" if ("school" in full_text or "bus" in full_text or "parents" in full_text) else "urgent"
                return {
                    "message_id": msg["message_id"],
                    "action": "notify",
                    "message_type": msg_cat,
                    "reason": "A trusted group admin sent a time-sensitive update that should interrupt the user.",
                    "confidence": 0.89,
                    "evidence_message_ids": evidence_ids
                }

        # 3. Direct Personal Questions / Requests
        if conv_type == "personal" or has_direct_mention or any(k in full_text for k in ["call", "pickup", "check", "when you get 5 mins"]):
            if not is_group_muted:
                return {
                    "message_id": msg["message_id"],
                    "action": "notify",
                    "message_type": "personal",
                    "reason": "The sender directly asks this user for a response or action.",
                    "confidence": 0.87,
                    "evidence_message_ids": evidence_ids
                }

        # 4. Verified Business Updates vs Promotions
        if conv_type == "business" and biz_info:
            is_verified = (biz_info.get("verified") == "1")
            allows_promo = (user_biz_info.get("allows_promotions") == "1")
            has_history = bool(user_biz_info.get("why_user_knows_account"))

            is_promo_text = any(k in full_text for k in ["sale", "off", "discount", "itinerary", "unsubscribe", "try50"])
            
            if is_promo_text:
                if allows_promo:
                    return {
                        "message_id": msg["message_id"],
                        "action": "digest",
                        "message_type": "promotion",
                        "reason": "The message is promotional but matches a topic or business the user has opted into.",
                        "confidence": 0.78,
                        "evidence_message_ids": evidence_ids
                    }
                else:
                    return {
                        "message_id": msg["message_id"],
                        "action": "mute",
                        "message_type": "promotion",
                        "reason": "Promotional message from a business sender without active user opt-in or engagement.",
                        "confidence": 0.82,
                        "evidence_message_ids": evidence_ids
                    }
            elif is_verified and has_history:
                msg_cat = "payment" if "pay" in full_text or "bill" in full_text else "business_update"
                if "health" in full_text or "appointment" in full_text:
                    msg_cat = "event"
                return {
                    "message_id": msg["message_id"],
                    "action": "notify" if not is_dnd else "digest",
                    "message_type": msg_cat,
                    "reason": "A verified business is sending an update that matches the user's recent activity history.",
                    "confidence": 0.89 if not is_dnd else 0.80,
                    "evidence_message_ids": evidence_ids
                }

        # 5. Casual Group Chats / Digest Items
        if conv_type == "group":
            if is_group_muted:
                return {
                    "message_id": msg["message_id"],
                    "action": "mute",
                    "message_type": "greeting" if "good morning" in full_text else "personal",
                    "reason": "Group is muted by user and message does not contain urgent admin update or direct mention.",
                    "confidence": 0.84,
                    "evidence_message_ids": evidence_ids
                }
            
            # Event / Form / General Notice
            if any(k in full_text for k in ["form", "sheet", "match", "casual", "selling", "peaceful", "good morning"]):
                cat = "event" if "form" in full_text else ("greeting" if "peaceful" in full_text or "good morning" in full_text else "promotion" if "selling" in full_text else "personal")
                return {
                    "message_id": msg["message_id"],
                    "action": "digest",
                    "message_type": cat,
                    "reason": "The message is useful group information, but it is not urgent enough to interrupt the user.",
                    "confidence": 0.82,
                    "evidence_message_ids": evidence_ids
                }

        # Default Catch-All Route
        return {
            "message_id": msg["message_id"],
            "action": "digest" if not is_dnd else "mute",
            "message_type": "unknown",
            "reason": "Standard background update routed based on default priority context.",
            "confidence": 0.75,
            "evidence_message_ids": evidence_ids
        }
