import os
import csv
import sys
from pathlib import Path

# Add code directory to sys.path
code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from context import ContextEngine
from media import HybridMediaProcessor
from retrieval import EvidenceRetriever
from router import NotificationRouter
from postprocess import OutputPostProcessor

def main():
    # 1. Determine dataset directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    dataset_dir = repo_root / "dataset"

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        sys.exit(1)

    print("==================================================")
    print("WhatsApp Message Notification Router - Initializing")
    print("==================================================")

    # 2. Initialize Pipeline Modules
    print("[1/5] Loading Context Engine...")
    context = ContextEngine(str(dataset_dir))

    print("[2/5] Initializing Hybrid Media Processor (VLM/OCR + Caching)...")
    media = HybridMediaProcessor(cache_dir=str(repo_root / ".cache"))

    print("[3/5] Initializing Evidence Retrieval Engine (RAG)...")
    retriever = EvidenceRetriever(str(dataset_dir))

    print("[4/5] Initializing Fenced Decision Router...")
    router = NotificationRouter(context, media, retriever)
    postprocessor = OutputPostProcessor()

    # 3. Read incoming messages
    messages_path = dataset_dir / "messages.csv"
    if not messages_path.exists():
        print(f"Error: messages.csv not found at {messages_path}")
        sys.exit(1)

    messages = []
    with open(messages_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append(row)

    print(f"[5/5] Processing {len(messages)} incoming messages...")

    predictions = []
    counts = {"notify": 0, "digest": 0, "mute": 0}

    for msg in messages:
        raw_pred = router.route_message(msg)
        clean_pred = postprocessor.process(raw_pred)
        predictions.append(clean_pred)
        counts[clean_pred["action"]] += 1

    # 4. Export predictions to dataset/output.csv
    output_path = dataset_dir / "output.csv"
    fieldnames = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print("\n==================================================")
    print(f"SUCCESS: Pipeline completed successfully!")
    print(f"Output saved to: {output_path}")
    print(f"Summary: Total={len(predictions)} | notify={counts['notify']} | digest={counts['digest']} | mute={counts['mute']}")
    print("==================================================")

if __name__ == "__main__":
    main()
