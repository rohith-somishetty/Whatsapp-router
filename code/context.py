import csv
import os
from datetime import datetime, time
from typing import Dict, Any, Optional

class ContextEngine:
    """
    Context Engine for indexing and performing fast O(1) lookups across all relational dataset files.
    """
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.users: Dict[str, Dict[str, Any]] = {}
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.group_members: Dict[tuple, Dict[str, Any]] = {}  # (group_id, user_id) -> data
        self.business_accounts: Dict[str, Dict[str, Any]] = {}
        self.user_business_history: Dict[tuple, Dict[str, Any]] = {}  # (user_id, business_id) -> data
        self.daily_summary: Dict[tuple, Dict[str, Any]] = {}  # (user_id, date_str) -> data
        self.images: Dict[str, str] = {}  # image_id -> file_path
        self.voice_notes: Dict[str, str] = {}  # voice_note_id -> file_path

        self._load_all()

    def _load_all(self):
        # 1. Load users
        users_path = os.path.join(self.dataset_dir, "users.csv")
        if os.path.exists(users_path):
            with open(users_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.users[row["user_id"]] = row

        # 2. Load groups
        groups_path = os.path.join(self.dataset_dir, "groups.csv")
        if os.path.exists(groups_path):
            with open(groups_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.groups[row["group_id"]] = row

        # 3. Load group members
        gm_path = os.path.join(self.dataset_dir, "group_members.csv")
        if os.path.exists(gm_path):
            with open(gm_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.group_members[(row["group_id"], row["user_id"])] = row

        # 4. Load business accounts
        biz_path = os.path.join(self.dataset_dir, "business_accounts.csv")
        if os.path.exists(biz_path):
            with open(biz_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.business_accounts[row["business_id"]] = row

        # 5. Load user business history
        ubh_path = os.path.join(self.dataset_dir, "user_business_history.csv")
        if os.path.exists(ubh_path):
            with open(ubh_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.user_business_history[(row["user_id"], row["business_id"])] = row

        # 6. Load daily notification summary
        dns_path = os.path.join(self.dataset_dir, "daily_notification_summary.csv")
        if os.path.exists(dns_path):
            with open(dns_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # key by user_id, date
                    date_key = row.get("date", row.get("created_at", ""))[:10]
                    self.daily_summary[(row["user_id"], date_key)] = row

        # 7. Load images
        img_path = os.path.join(self.dataset_dir, "images.csv")
        if os.path.exists(img_path):
            with open(img_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.images[row["image_id"]] = row["file_path"]

        # 8. Load voice notes
        vn_path = os.path.join(self.dataset_dir, "voice_notes.csv")
        if os.path.exists(vn_path):
            with open(vn_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.voice_notes[row["voice_note_id"]] = row["file_path"]

    def is_quiet_hours(self, user_id: str, timestamp_str: str) -> bool:
        """
        Determines whether the given timestamp falls inside the user's do_not_disturb_window.
        Format in CSV: e.g. "22:00-07:00" or "21:30-07:30".
        """
        user_info = self.users.get(user_id)
        if not user_info or not user_info.get("do_not_disturb_window"):
            return False

        dnd_str = user_info["do_not_disturb_window"].strip()
        if "-" not in dnd_str:
            return False

        try:
            start_str, end_str = dnd_str.split("-")
            start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_time = datetime.strptime(end_str.strip(), "%H:%M").time()

            # Parse input timestamp
            msg_dt = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M:%S" if len(timestamp_str) > 16 else "%Y-%m-%d %H:%M")
            msg_time = msg_dt.time()

            if start_time <= end_time:
                return start_time <= msg_time <= end_time
            else:
                # Overnight DND window (e.g., 22:00 to 07:00)
                return msg_time >= start_time or msg_time <= end_time
        except Exception:
            return False

    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        return self.users.get(user_id, {})

    def get_group_info(self, group_id: str) -> Dict[str, Any]:
        return self.groups.get(group_id, {})

    def get_group_membership(self, group_id: str, user_id: str) -> Dict[str, Any]:
        return self.group_members.get((group_id, user_id), {})

    def get_business_info(self, business_id: str) -> Dict[str, Any]:
        return self.business_accounts.get(business_id, {})

    def get_user_business_history(self, user_id: str, business_id: str) -> Dict[str, Any]:
        return self.user_business_history.get((user_id, business_id), {})

    def get_image_path(self, image_id: str) -> Optional[str]:
        rel_path = self.images.get(image_id)
        if rel_path:
            return os.path.join(self.dataset_dir, rel_path)
        return None

    def get_voice_note_path(self, voice_note_id: str) -> Optional[str]:
        rel_path = self.voice_notes.get(voice_note_id)
        if rel_path:
            return os.path.join(self.dataset_dir, rel_path)
        return None
