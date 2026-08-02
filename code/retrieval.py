import os
import csv
import math
from datetime import datetime
from typing import Dict, List, Any

class EvidenceRetriever:
    """
    Historical Evidence Retrieval Engine.
    Uses hybrid relational filtering, lexical/topic keyword matching, user reaction events,
    and temporal decay scoring to retrieve top-K evidence message IDs.
    """
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.history: List[Dict[str, Any]] = []
        self.events: Dict[str, Dict[str, Any]] = {}  # message_id -> event_data
        self._load_data()

    def _load_data(self):
        # Load message_events.csv
        events_path = os.path.join(self.dataset_dir, "message_events.csv")
        if os.path.exists(events_path):
            with open(events_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.events[row["message_id"]] = row

        # Load message_history.csv
        history_path = os.path.join(self.dataset_dir, "message_history.csv")
        if os.path.exists(history_path):
            with open(history_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.history.append(row)

    def _tokenize(self, text: str) -> set:
        if not text:
            return set()
        words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
        return {w for w in words if len(w) > 2}

    def find_evidence(self, msg: Dict[str, Any], max_k: int = 2) -> str:
        """
        Finds top-K historical evidence message IDs matching the given incoming message context.
        Returns semicolon-separated message_ids string (e.g. "message_0013;message_0014") or "none".
        """
        user_id = msg.get("user_id", "")
        group_id = msg.get("group_id", "")
        business_id = msg.get("business_id", "")
        sender_user_id = msg.get("sender_user_id", "")
        msg_text = msg.get("message_text", "")
        media_text = msg.get("media_text", "")
        combined_text = f"{msg_text} {media_text}".strip()
        msg_tokens = self._tokenize(combined_text)

        msg_created_at = msg.get("created_at", "")
        try:
            msg_dt = datetime.strptime(msg_created_at[:16], "%Y-%m-%d %H:%M")
        except Exception:
            msg_dt = datetime.now()

        candidates = []

        for h in self.history:
            # Must belong to the same receiving user
            if h.get("user_id") != user_id:
                continue

            score = 0.0

            # 1. Relational Matching
            h_group = h.get("group_id", "")
            h_biz = h.get("business_id", "")
            h_sender = h.get("sender_user_id", "")

            if group_id and h_group == group_id:
                score += 3.0
            if business_id and h_biz == business_id:
                score += 3.5
            if sender_user_id and h_sender == sender_user_id:
                score += 3.0

            # 2. Text Keyword Match
            h_text = h.get("message_text", "")
            h_tokens = self._tokenize(h_text)
            overlap = len(msg_tokens.intersection(h_tokens))
            if overlap > 0:
                score += overlap * 1.5

            # 3. Reaction Event Boost
            h_id = h.get("message_id", "")
            event_info = self.events.get(h_id, {})
            if event_info:
                if event_info.get("muted_after_message") == "1" or event_info.get("message_reported") == "1":
                    score += 2.5
                elif event_info.get("notification_dismissed") == "1":
                    score += 1.5
                elif event_info.get("message_replied") == "1":
                    score += 2.0

            # 4. Forwarded count pattern match
            if int(msg.get("forwarded_count", 0)) > 0 and int(h.get("forwarded_count", 0)) > 0:
                score += 2.0

            if score < 2.5:
                continue

            # 5. Temporal Decay
            h_created_at = h.get("created_at", "")
            try:
                h_dt = datetime.strptime(h_created_at[:16], "%Y-%m-%d %H:%M")
                days_diff = abs((msg_dt - h_dt).total_seconds()) / 86400.0
            except Exception:
                days_diff = 30.0

            decay_multiplier = math.exp(-0.01 * days_diff)
            final_score = score * decay_multiplier

            candidates.append((final_score, h_id))

        if not candidates:
            return "none"

        # Sort descending by final_score
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_ids = [cand[1] for cand in candidates[:max_k]]
        
        return ";".join(top_ids) if top_ids else "none"
