import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticContextClassifier:
    """
    Consolidated Feature Extraction & Multi-Centroid BERT Contextual Classifier.
    Includes state-progression signals, structural co-occurrences, and soft-downweighting.
    """
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Multi-sentence anchor clusters per category for multi-centroid vector averaging
        self.categories_clusters = {
            "urgent": [
                "family medical emergency hospital admitted room accident collapsed doctor ICU emergency call back immediately",
                "car broke down expressway towing service pick me up roadside assistance vehicle disabled",
                "production server API gateway failing 500 error critical outage database locked memory leak root breach bug patch ASAP",
                "fire alarm building evacuate stairs water pipe main line burst emergency flooding",
                "lost passport wallet airport emergency help contacting embassy lost money taxi fare urgent",
                "school infirmary called child high fever emergency pick up right now",
                "vitals unstable ICU ward ambulance call immediately police report stolen vehicle flight schedule changed",
                "water supply tanker leaving in 20 mins fill water now",
                "bus leaving 15 mins early road blocked driver waiting",
                "retry count crossed alert threshold escalation starts in 20 minutes"
            ],
            "scam": [
                "phishing link domain mismatch unverified imposter bank account blocked verify credentials suspended",
                "congratulations won bitcoin BTC crypto giveaway claim reward lucky draw gift voucher spin wheel free prize",
                "work from home typing online part time job earn daily no experience join telegram bit.ly link",
                "fake tax refund income tax department account details instant cash loan app 0 interest pre-approved",
                "apk download whatsapp gold edition custom theme 50g free internet data recharge free TRAI sim deactivation",
                "accidentally sent money UPI refund claim scratch card reward casino bonus double cash"
            ],
            "forward": [
                "forward this message to friends within minutes or account deleted bad luck 7 years devotional chain prayer",
                "drink warm water with lemon cure all diseases forward to save lives send to family members",
                "send this blessings video to contacts forward devotional image"
            ],
            "payment": [
                "bank account transaction salary credited debited credit card bill due date split share dinner UPI money transfer",
                "rent payment receipt tuition fee maintenance charge GPay alert salary credited account statement balance",
                "Kite alert order executed bought shares subscription renewed bill payment confirmation"
            ],
            "event": [
                "invitation invited party concert tickets picnic reunion wedding save the date book of the month discussion",
                "society AGM meeting water supply shut off notice parking sticker distribution yoga session terrace garden",
                "annual family reunion picnic concert ticket gate opens resort booking",
                "health update appointment prescription claim pickup ready",
                "cultural night form open add flat number dish in sheet",
                "school circular attached check timing consent note"
            ],
            "promotion": [
                "exclusive discount coupon end of season sale 70 percent off top brands fashion catalog flash sale alert wishlist items",
                "big savings day up to 80 percent off electronics pink friday sale",
                "selling cycle helmet medium size pickup near gate",
                "welcome get 50 percent off with TRY50 code",
                "shopping offer available extra discounts today"
            ],
            "greeting": [
                "good morning good evening happy birthday happy anniversary congratulations hope you are having a great week catch up soon coffee",
                "safe flight wish reach home safely thanks for helping out excited for your new house",
                "happy diwali happy new year society members",
                "stay positive keep smiling share blessings with everyone"
            ],
            "personal": [
                "casual chat friends dinner plans Saturday restaurant recommendation recipe lasagna trip photos movie review",
                "DIY gardening video tennis match podcast productivity hiking trail coordinates hotel link room booking",
                "family reunion lunch check in how job is going",
                "when you get 5 mins can you call checking if Sunday pickup works",
                "anyone watching match tonight start score thread after dinner",
                "volunteer sheet coordinating registrations for Saturday"
            ],
            "business_update": [
                "order shipped dispatched out for delivery package tracking update statement ready view account balance update appointment reminder",
                "order packed expected to reach local hub check delivery details",
                "safety advisory brand says never ask for OTP or payment details"
            ]
        }

        self.cat_names = list(self.categories_clusters.keys())
        
        # Compute multi-centroid mean embedding for each category
        self.anchor_centroids = []
        for cat in self.cat_names:
            sentences = self.categories_clusters[cat]
            embeddings = self.model.encode(sentences)
            centroid = np.mean(embeddings, axis=0)
            self.anchor_centroids.append(centroid)
            
        self.anchor_centroids = np.array(self.anchor_centroids)

    def extract_structural_features(self, text: str, is_admin: bool = False, has_mention: bool = False, conv_type: str = "personal") -> dict:
        """
        Consolidated structural pattern extractor.
        """
        clean_text = text.lower()

        # Fix 1: Generalized Admin + Relative Time Phrasing
        time_rel_pattern = r"\b\d+\s*(?:mins?|minutes?|hours?|hrs?)\b"
        action_verb_pattern = r"\b(?:wait|leave|leaving|close|closing|hurry|last call|deadline|expire|expires|cutoff|due|max)\b"
        has_time_bound = bool(re.search(time_rel_pattern, clean_text) and re.search(action_verb_pattern, clean_text))
        is_admin_time_urgent = is_admin and has_time_bound and (conv_type == "group")

        # Fix 2 Revised: Computed Status-Progression Feature (State-change verb + Near-term time reference)
        state_change_pattern = r"\b(?:packed|shipped|dispatched|out for delivery|in transit|ready|due|arriving|attached|open|collected|delivered)\b"
        near_term_time_pattern = r"\b(?:today|tomorrow|tonight|this evening|eod|by\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|in\s+\d+\s*(?:hours?|hrs?|mins?|minutes?))\b"
        has_status_progression = bool(re.search(state_change_pattern, clean_text) and re.search(near_term_time_pattern, clean_text))
        
        # Check if genuinely time-critical (e.g. arriving in 10 minutes)
        is_critical_time_arrival = bool(re.search(r"\b(?:arriving|reaching|delivery)\s+(?:in|by)\s+\d+\s*(?:mins?|minutes?)\b", clean_text))

        # Fix 3: Generalized C2C Selling (Structural co-occurrence required)
        selling_verb_pattern = r"\b(?:selling|for sale|wts|available for sale|moving out sale)\b"
        price_logistics_pattern = r"(?:(?:₹|\$|rs\.?|inr)\s*\d+|\b\d+\s*(?:k|bucks)\b|\b(?:pickup|pick-up|dm for price|gate \d+|self pick|collection|shipping|condition)\b)"
        is_c2c_selling = bool(re.search(selling_verb_pattern, clean_text) and re.search(price_logistics_pattern, clean_text))

        # Fix 4: @Mention + Broad Escalation Set
        escalation_vocab_pattern = r"\b(?:outage|blocked|p0|p1|incident|down|hotfix|breach|failing|failure|crash|crashed|latency spike|sev-1|sev-0|escalat(?:e|ion)|out of memory|ootm|prod(?:uction)? review|failed-payment|queue number)\b"
        is_mention_escalation = has_mention and bool(re.search(escalation_vocab_pattern, clean_text)) and (conv_type in ["group", "business"])

        # Fix 2: Soft Negation Check
        negation_phrases = [
            r"\bnothing urgent\b", r"\bnot urgent\b", r"\bno emergency\b",
            r"\bdon'?t call\b", r"\bdo not call\b", r"\bno pressure\b",
            r"\bwhen you get time\b", r"\bno need to rush\b", r"\btalk tomorrow\b"
        ]
        has_soft_negation = any(re.search(pat, clean_text) for pat in negation_phrases)

        return {
            "is_admin_time_urgent": is_admin_time_urgent,
            "has_status_progression": has_status_progression,
            "is_critical_time_arrival": is_critical_time_arrival,
            "is_c2c_selling": is_c2c_selling,
            "is_mention_escalation": is_mention_escalation,
            "has_soft_negation": has_soft_negation
        }

    def classify_context(self, text: str, is_admin: bool = False, has_mention: bool = False, conv_type: str = "personal"):
        if not text.strip():
            return "personal", 0.5

        clean_text = re.sub(r'http\S+|www\S+', '', text.lower())
        
        # Structural Feature Extraction
        struct_feats = self.extract_structural_features(text, is_admin, has_mention, conv_type)

        # C2C Selling Structural Override
        if struct_feats["is_c2c_selling"]:
            return "promotion", 0.88

        # Admin Time-Bound Urgency Override
        if struct_feats["is_admin_time_urgent"] or struct_feats["is_mention_escalation"]:
            return "urgent", 0.92

        # Compute BERT Vector Embeddings
        text_emb = self.model.encode([clean_text])
        sims = cosine_similarity(text_emb, self.anchor_centroids)[0]

        # Soft Downweighting for Negation
        if struct_feats["has_soft_negation"]:
            urgent_idx = self.cat_names.index("urgent") if "urgent" in self.cat_names else -1
            if urgent_idx != -1:
                sims[urgent_idx] *= 0.25

        best_idx = np.argmax(sims)
        best_cat = self.cat_names[best_idx]
        best_score = float(sims[best_idx])

        return best_cat, best_score
