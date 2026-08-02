import os
import csv
import sys
from pathlib import Path

def evaluate():
    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "dataset"
    output_path = dataset_dir / "output.csv"
    sample_path = dataset_dir / "sample_messages.csv"
    messages_path = dataset_dir / "messages.csv"

    print("==================================================")
    print("WhatsApp Router Benchmark Evaluator")
    print("==================================================")

    if not output_path.exists():
        print(f"FAIL: output.csv not found at {output_path}")
        sys.exit(1)

    # 1. Read Output
    output_rows = []
    with open(output_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        output_rows = list(reader)

    # 2. Column Schema Check
    required_cols = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
    if not output_rows:
        print("FAIL: output.csv is empty!")
        sys.exit(1)

    actual_cols = list(output_rows[0].keys())
    if actual_cols != required_cols:
        print(f"FAIL: Columns mismatch. Expected {required_cols}, got {actual_cols}")
        sys.exit(1)

    print("[OK] Schema Check Passed: Column header matches required format.")

    # 3. Enum & Data Validation
    allowed_actions = {"notify", "digest", "mute"}
    allowed_types = {
        "personal", "urgent", "event", "payment", "business_update",
        "promotion", "greeting", "forward", "spam", "scam", "unknown"
    }

    errors = 0
    for idx, row in enumerate(output_rows):
        if row["action"] not in allowed_actions:
            print(f"Row {idx+1}: Invalid action '{row['action']}'")
            errors += 1
        if row["message_type"] not in allowed_types:
            print(f"Row {idx+1}: Invalid message_type '{row['message_type']}'")
            errors += 1
        try:
            conf = float(row["confidence"])
            if not (0.0 <= conf <= 1.0):
                print(f"Row {idx+1}: Confidence {conf} out of range [0, 1]")
                errors += 1
        except ValueError:
            print(f"Row {idx+1}: Invalid numeric confidence '{row['confidence']}'")
            errors += 1

    if errors == 0:
        print("[OK] Enum & Range Check Passed: All actions, types, and confidences are valid.")
    else:
        print(f"FAIL: Found {errors} validation errors in output.csv.")
        sys.exit(1)

    # 4. Compare against Sample Messages Benchmark (if available)
    if sample_path.exists():
        sample_rows = {}
        with open(sample_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_rows[row["message_id"]] = row

        match_action = 0
        match_type = 0
        total_samples = 0

        for row in output_rows:
            msg_id = row["message_id"]
            if msg_id in sample_rows:
                total_samples += 1
                gt = sample_rows[msg_id]
                if row["action"] == gt["action"]:
                    match_action += 1
                if row["message_type"] == gt["message_type"]:
                    match_type += 1

        if total_samples > 0:
            action_acc = (match_action / total_samples) * 100
            type_acc = (match_type / total_samples) * 100
            print("\n--- Benchmark Validation Metrics ---")
            print(f"Sample Rows Evaluated: {total_samples}")
            print(f"Action Accuracy:       {action_acc:.1f}% ({match_action}/{total_samples})")
            print(f"Message Type Accuracy: {type_acc:.1f}% ({match_type}/{total_samples})")

    print("\n==================================================")
    print("SUCCESS: Benchmark Evaluation Passed! All checks green.")
    print("==================================================")

if __name__ == "__main__":
    evaluate()
