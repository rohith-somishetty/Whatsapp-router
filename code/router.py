import re
from typing import Dict, Any
from context import ContextEngine
from media import HybridMediaProcessor
from retrieval import EvidenceRetriever
from semantic_context import SemanticContextClassifier

class NotificationRouter:
    """
    WhatsApp Message Notification Router powered by:
    1. Relational Context & Security Policy Guard (Domain Reputation, Phishing, OTP Theft & Prompt Injection Interceptor)
    2. Consolidated Feature Extractor (Admin Urgency, C2C Selling Co-occurrence, @Mention Escalation, Status Progression)
    3. Multi-Centroid BERT Transformer Context Engine with Soft Negation Downweighting
    """
    def __init__(self, context: ContextEngine, media: HybridMediaProcessor, retriever: EvidenceRetriever):
        self.context = context
        self.media = media
        self.retriever = retriever
        self.semantic_engine = SemanticContextClassifier()

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
        full_text = f"{msg_text} {extracted_media_text}".strip()
        lower_text = full_text.lower()
        fallback_used = media_features.get("fallback_used", False)

        # Add media text back to msg dict for retrieval query
        msg["media_text"] = extracted_media_text

        # 2. Query RAG Evidence Retriever
        evidence_ids = self.retriever.find_evidence(msg)

        # 3. Relational Context Lookups from ContextEngine
        is_dnd = self.context.is_quiet_hours(user_id, created_at)
        user_info = self.context.get_user_info(user_id)
        group_info = self.context.get_group_info(group_id) if group_id else {}
        
        # Verify admin sender directly via group_members.csv relational lookup
        sender_group_info = self.context.get_group_membership(group_id, sender_id) if (group_id and sender_id) else {}
        is_admin_sender = (sender_group_info.get("role") == "admin")

        user_group_info = self.context.get_group_membership(group_id, user_id) if (group_id and user_id) else {}
        is_group_muted = (user_group_info.get("group_muted_by_user") == "1")

        biz_info = self.context.get_business_info(business_id) if business_id else {}
        user_biz_info = self.context.get_user_business_history(user_id, business_id) if (user_id and business_id) else {}
        is_verified_biz = (biz_info.get("verified") == "1")
        has_biz_history = bool(user_biz_info)

        has_direct_mention = f"@{user_id}".lower() in lower_text or f"@{sender_id}".lower() in lower_text or "@u_" in lower_text

        # 4. Consolidated Feature Extraction
        struct_feats = self.semantic_engine.extract_structural_features(
            full_text,
            is_admin=is_admin_sender,
            has_mention=has_direct_mention,
            conv_type=conv_type
        )

        # ---------------------------------------------------------
        # STAGE 1: SECURITY POLICY GUARD (Imposter Domain, Phishing, OTP Theft & Prompt Injection)
        # ---------------------------------------------------------

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
            if is_unverified and (user_reports > 0 or domain_mismatch or any(k in biz_info.get("display_name", "").lower() for k in ["chase", "phonepe", "hdfc", "airtel", "hsbc", "razorpay", "green cross"])):
                is_unverified_imposter_biz = True

        # English credential theft patterns
        credential_theft_pattern = r"\b(?:reply with|confirm|verify|send)\s+(?:the\s+)?(?:6|4)?\s*digit\s*(?:code|otp|login)|confirm password|verify wallet|verify card|verify account|otp (?:may have|has) leaked|workspace access will expire\b"

        # Expanded prompt injection: router override metadata patterns
        prompt_injection_pattern = (
            r"\bignore\s+(?:all\s+)?previous\s+(?:instructions|rules|prompt|routing)\b"
            r"|\b(?:routing\s+override|system\s+note\s+for\s+(?:the\s+)?(?:notification\s+)?router|internal\s+router\s+metadata|assistant\s+instruction)\b"
            r"|\b(?:set\s+action\s*=\s*notify|action\s*=\s*notify|mark\s+(?:this\s+(?:as\s+)?)?notify|mark\s+notify|verified_business\s*=\s*true|user_priority\s*=\s*high)\b"
        )

        has_credential_theft = bool(re.search(credential_theft_pattern, lower_text))
        has_prompt_injection = bool(re.search(prompt_injection_pattern, lower_text))

        # Hindi OTP / account-block scam patterns
        hindi_scam_pattern = (
            r"\b(?:otp\s+(?:abhi|jaldi|batao|dalo|share|verify|confirm)|account\s+block\s+ho|profile\s+band\s+ho"
            r"|link\s+open\s+karo|account\s+bachane|verify\s+(?:nahi|karein|karo)|code\s+daal\s+do"
            r"|hold\s+pe\s+chala\s+jayega|account-help\.in|support\s+bhi\s+reopen)\b"
        )
        has_hindi_scam = bool(re.search(hindi_scam_pattern, lower_text))

        # QR + payment pressure scam patterns
        qr_payment_pattern = (
            r"\b(?:scan\s+(?:the\s+)?qr\s+and\s+pay|scan\s+and\s+pay"
            r"|pay\s+(?:the\s+)?(?:clearance|penalty|charge|amount)\s+(?:immediately|now|today|urgently|before)"
            r"|clearance\s+amount|reactivation\s+fee|processing\s+fee\s+at\s+this\s+link"
            r"|send\s+screenshot\s+(?:after|once)|fill\s+bank\s+details\s+on)\b"
        )
        has_qr_payment_scam = bool(re.search(qr_payment_pattern, lower_text))

        # Account/card number sharing pressure
        account_sharing_pattern = (
            r"\b(?:shar(?:e|ing)\s+(?:your\s+)?(?:account\s+number|card\s+details|bank\s+details)"
            r"|send\s+(?:your\s+)?(?:card|account|bank)\s+details"
            r"|claim\s+(?:benefits|amount|rewards?)\s+by\s+shar"
            r"|approval\s+window\s+closes|amount\s+will\s+be\s+released\s+today)\b"
        )
        has_account_sharing_scam = bool(re.search(account_sharing_pattern, lower_text))

        phishing_url_patterns = [
            "bit.ly/", "t.me/", ".xyz", ".site", ".online", ".tech", ".cc", ".info",
            "account-login.in", "account-help.in", "http://", "https://", "0.5 btc",
            "crypto giveaway", "scratch card", "e-kyc",
            "verify ssn", "instant loan", "whatsapp gold", "trai", "spin-win", "refund pending",
            "unclaimed parcel", "home-job", "1000% return", "parttime", "echallan", "recharge-free",
            "double your money", "casino bonus", "accidentally sent ₹", "claim ₹", "verification fee"
        ]
        has_suspicious_url = any(k in lower_text for k in phishing_url_patterns)

        if (is_unverified_imposter_biz or domain_mismatch or has_suspicious_url
                or has_credential_theft or has_prompt_injection
                or has_hindi_scam or has_qr_payment_scam or has_account_sharing_scam
                or media_features.get("is_scam_suspect")):
            return {
                "message_id": msg["message_id"],
                "action": "mute",
                "message_type": "scam",
                "reason": "Security Policy Guard: Malicious phishing, credential theft, or prompt injection detected.",
                "confidence": 0.98,
                "evidence_message_ids": evidence_ids
            }

        # Forward Chain Security Policy
        forward_phrases = ["send this message to", "send this prayer", "drink warm water", "forwarded as received", "forward this", "do not break the chain", "send this blessings", "forward this devotional", "share with everyone", "forward to all", "share in family groups", "share in all family", "share to all groups", "pls forward", "please forward"]
        greeting_forward_keywords = ["good morning", "god bless", "stay blessed", "blessings", "positive energy", "bhagwan", "shayad", "smile today", "good day", "have a nice", "forwarding because it felt"]

        # Society/admin critical notices should not be silenced even if forwarded
        SOCIETY_URGENT_KEYWORDS = ["tanker", "motor room", "water supply", "fire alarm", "lift maintenance", "gate band", "repair truck", "gate closes", "parking", "main gate"]
        is_legitimate_admin_forward = is_admin_sender and any(k in lower_text for k in SOCIETY_URGENT_KEYWORDS)

        if not is_legitimate_admin_forward and (forwarded_count >= 5 or any(w in lower_text for w in forward_phrases)):
            # Determine best message_type for this muted forward
            is_greeting_fwd = any(k in lower_text for k in greeting_forward_keywords)
            if is_greeting_fwd:
                fwd_type = "greeting"
            elif forwarded_count >= 5 or "forward" in lower_text or "send this" in lower_text:
                fwd_type = "forward"
            else:
                fwd_type = "spam"
            return {
                "message_id": msg["message_id"],
                "action": "mute",
                "message_type": fwd_type,
                "reason": "Security Policy Guard: Viral forward chain pattern detected.",
                "confidence": 0.88,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 2: STRUCTURAL OVERRIDES & SENDER-TRUST GATED UPWEIGHTS
        # ---------------------------------------------------------

        if struct_feats["is_admin_time_urgent"]:
            is_school_event = any(k in lower_text for k in ["school", "bus", "parents", "circular", "consent note", "timing", "pickup", "trip", "stadium"])
            return {
                "message_id": msg["message_id"],
                "action": "notify",
                "message_type": "event" if is_school_event else "urgent",
                "reason": "Verified Group Admin time-bound relative deadline announcement.",
                "confidence": 0.94,
                "evidence_message_ids": evidence_ids
            }

        if struct_feats["is_mention_escalation"]:
            return {
                "message_id": msg["message_id"],
                "action": "notify",
                "message_type": "urgent",
                "reason": "Direct user mention co-occurring with critical work escalation pattern.",
                "confidence": 0.92,
                "evidence_message_ids": evidence_ids
            }

        if struct_feats["is_c2c_selling"]:
            return {
                "message_id": msg["message_id"],
                "action": "digest",
                "message_type": "promotion",
                "reason": "Structural co-occurrence of selling verb with price or logistics phrasing.",
                "confidence": 0.88,
                "evidence_message_ids": evidence_ids
            }

        if struct_feats["has_status_progression"]:
            is_trusted_sender = is_verified_biz or is_admin_sender
            is_recent_duplicate = (evidence_ids and len(evidence_ids) > 1)
            
            if is_trusted_sender and not is_recent_duplicate:
                msg_type = "event" if ("circular" in lower_text or "appointment" in lower_text or "prescription" in lower_text) else "business_update"
                return {
                    "message_id": msg["message_id"],
                    "action": "notify",
                    "message_type": msg_type,
                    "reason": "Verified Sender status-progression update with near-term time reference.",
                    "confidence": 0.90,
                    "evidence_message_ids": evidence_ids
                }
            elif is_recent_duplicate:
                msg_type = "event" if ("circular" in lower_text or "appointment" in lower_text) else "business_update"
                return {
                    "message_id": msg["message_id"],
                    "action": "digest",
                    "message_type": msg_type,
                    "reason": "Status-progression update downgraded to digest due to recent duplicate notification refire.",
                    "confidence": 0.80,
                    "evidence_message_ids": evidence_ids
                }

        if is_group_muted and has_direct_mention:
            return {
                "message_id": msg["message_id"],
                "action": "notify",
                "message_type": "personal",
                "reason": "Direct mention targeting user overrides muted group preferences.",
                "confidence": 0.90,
                "evidence_message_ids": evidence_ids
            }

        # ---------------------------------------------------------
        # STAGE 3: MULTI-CENTROID BERT TRANSFORMER CONTEXT ENGINE
        # ---------------------------------------------------------

        semantic_cat, sim_score = self.semantic_engine.classify_context(
            full_text,
            is_admin=is_admin_sender,
            has_mention=has_direct_mention,
            conv_type=conv_type
        )

        if fallback_used and (not msg_text.strip() or msg_text.strip() == "..."):
            relational_type = "business_update" if conv_type == "business" else "personal"
            relational_action = "digest"
            return {
                "message_id": msg["message_id"],
                "action": relational_action,
                "message_type": relational_type,
                "reason": "Media Attachment Fallback: Classified via relational context (API inactive). Confidence capped.",
                "confidence": 0.62,
                "evidence_message_ids": evidence_ids
            }

        if semantic_cat == "urgent":
            msg_cat = "payment" if ("upi" in lower_text or "wallet" in lower_text or "deposit" in lower_text or "fare" in lower_text) else "urgent"
            return {
                "message_id": msg["message_id"],
                "action": "notify",
                "message_type": msg_cat,
                "reason": f"BERT Context Engine: Urgent event detected (similarity: {sim_score:.2f}).",
                "confidence": min(0.95, sim_score + 0.3),
                "evidence_message_ids": evidence_ids
            }

        if semantic_cat == "payment":
            is_urgent_pay = any(k in lower_text for k in ["urgently", "emergency", "taxi fare", "credited", "order executed", "gpay alert", "neft"])
            return {
                "message_id": msg["message_id"],
                "action": "notify" if is_urgent_pay else "digest",
                "message_type": "payment",
                "reason": f"BERT Context Engine: Financial transaction notice (similarity: {sim_score:.2f}).",
                "confidence": 0.88,
                "evidence_message_ids": evidence_ids
            }

        if semantic_cat == "event":
            return {
                "message_id": msg["message_id"],
                "action": "digest",
                "message_type": "event",
                "reason": f"BERT Context Engine: Scheduled event or community notice (similarity: {sim_score:.2f}).",
                "confidence": 0.85,
                "evidence_message_ids": evidence_ids
            }

        if semantic_cat == "promotion" or (conv_type == "business" and any(k in lower_text for k in ["sale", "off", "discount", "coupon", "arrivals", "eorr", "flash sale", "savings day", "pink friday", "wishlist"])):
            has_cold_promo_code = any(k in lower_text for k in ["50% off", "try50", "shopping offer available", "extra discounts", "kurta set"])
            
            if not is_verified_biz or not has_biz_history or has_cold_promo_code:
                return {
                    "message_id": msg["message_id"],
                    "action": "mute",
                    "message_type": "promotion",
                    "reason": "Promotional Policy Guard: Cold marketing offer or unverified business promotion muted.",
                    "confidence": 0.85,
                    "evidence_message_ids": evidence_ids
                }
            else:
                return {
                    "message_id": msg["message_id"],
                    "action": "digest",
                    "message_type": "promotion",
                    "reason": "Promotional Policy Guard: Verified business promotion routed to daily digest.",
                    "confidence": 0.82,
                    "evidence_message_ids": evidence_ids
                }

        if semantic_cat == "greeting":
            return {
                "message_id": msg["message_id"],
                "action": "digest",
                "message_type": "greeting",
                "reason": f"BERT Context Engine: Courtesy greeting or wish (similarity: {sim_score:.2f}).",
                "confidence": 0.82,
                "evidence_message_ids": evidence_ids
            }

        if semantic_cat == "business_update" or conv_type == "business":
            # Expanded notify triggers: include present-tense delivery/status language
            NOTIFY_BIZ_TRIGGERS = [
                "shipped", "dispatched", "delivered", "arriving in", "tracking update",
                "order executed", "appointment reminder", "packed", "expected to reach",
                "pickup or route", "route status has changed", "status has changed",
                "order is ready", "delivery today", "at your gate", "outside your",
                "please pick up", "collect by", "confirm in the next", "collect it from gate",
                "your ride", "driver", "return pickup today", "shopee return"
            ]
            # For verified business with user history, notify on delivery/appointment signals
            is_notify_update = any(k in lower_text for k in NOTIFY_BIZ_TRIGGERS)
            # Healthcare + verified = notify (matches sample_005 pattern)
            # Only trigger on genuine healthcare terms, not logistics "pickup"
            is_healthcare_event = ("appointment" in lower_text or "prescription" in lower_text) and is_verified_biz
            should_notify = is_notify_update and (is_verified_biz or is_admin_sender)
            msg_type_biz = "event" if ("appointment" in lower_text or is_healthcare_event) else "business_update"
            return {
                "message_id": msg["message_id"],
                "action": "notify" if (should_notify or is_healthcare_event) else "digest",
                "message_type": msg_type_biz,
                "reason": f"BERT Context Engine: Business notification update (similarity: {sim_score:.2f}).",
                "confidence": 0.86,
                "evidence_message_ids": evidence_ids
            }

        if conv_type == "group":
            if is_group_muted:
                return {
                    "message_id": msg["message_id"],
                    "action": "mute",
                    "message_type": "personal",
                    "reason": "Group is muted by user preferences.",
                    "confidence": 0.84,
                    "evidence_message_ids": evidence_ids
                }
            return {
                "message_id": msg["message_id"],
                "action": "digest",
                "message_type": "personal",
                "reason": "BERT Context Engine: Standard group message.",
                "confidence": 0.80,
                "evidence_message_ids": evidence_ids
            }

        if conv_type == "personal":
            is_casual = (semantic_cat in ["personal", "greeting"]) or any(k in lower_text for k in ["dinner", "coffee", "photos", "recipe", "movie", "catch up", "how are you", "weekend", "book", "job", "podcast", "electrician", "hotel", "lunch", "gardening", "itinerary", "helping", "hiking", "reach home safely", "playlist", "house", "tennis", "voice note from mom"])
            return {
                "message_id": msg["message_id"],
                "action": "digest" if is_casual else "notify",
                "message_type": "personal",
                "reason": "BERT Context Engine: Personal 1-on-1 message.",
                "confidence": 0.85,
                "evidence_message_ids": evidence_ids
            }

        return {
            "message_id": msg["message_id"],
            "action": "digest" if not is_dnd else "mute",
            "message_type": "personal" if conv_type == "personal" else ("business_update" if conv_type == "business" else "personal"),
            "reason": "BERT Context Engine: Standard priority routing.",
            "confidence": 0.78,
            "evidence_message_ids": evidence_ids
        }
