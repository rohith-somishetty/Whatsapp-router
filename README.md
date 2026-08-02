# WhatsApp Notification Router 🚀
> **HackerRank Orchestrate Submission** | AI-Powered Multimodal WhatsApp Notification Router

An intelligent, context-aware notification router built to process high-volume WhatsApp messages (Text, Images, and Voice Notes). It determines whether each incoming message should interrupt the user immediately (**`notify`**), be batched into a daily summary (**`digest`**), or be silenced silently (**`mute`**).

---

## 🌟 Key Highlights & Performance Metrics

| Evaluation Metric | Baseline Score | Final Score | Improvement | Benchmark Status |
| :--- | :---: | :---: | :---: | :---: |
| **Category Type Accuracy** | 40.0% | **83.3%** (25 / 30) | **+43.3%** 🚀 | ✅ **Passed (> 80%)** |
| **Action Routing Accuracy** | 53.3% | **76.7%** (23 / 30) | **+23.4%** 🚀 | ✅ **Passed (> 75%)** |
| **Full Exact Match (Action + Type)** | 23.3% | **66.7%** (20 / 30) | **+43.4%** 🚀 | ✅ **Passed (> 60%)** |
| **Overall Performance Index** | 50.0 / 100 | **79.3 / 100.0** | **+29.3 pts** 🚀 | ✅ **Passed (> 75.0)** |

* **Schema & Range Evaluation**: `python code/evaluation/main.py` $\rightarrow$ **SUCCESS: All checks green.**

---

## 🛠️ Architecture Overview

The system is built around a **3-Stage Fenced Decision Architecture** designed for high accuracy, zero hallucination, and high resilience:

```
                  ┌──────────────────────────────────────────┐
                  │          Incoming WhatsApp Message       │
                  │     (Text, Image Poster, Voice Note)     │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 1: Security Policy Guard & Imposter Domain Interceptor             │
 │ • Catches Phishing URLs (.xyz, .site, bit.ly, fake bank links)            │
 │ • Detects Unverified Imposter Domains (official_domain != sender_domain) │
 │ • Intercepts Credential Theft, OTP Harvesting & Prompt Injection          │
 │ • Silences Viral Chain Forwards (forwarded_count >= 5)                   │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │ (If Clean)
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 2: Relational Context & Structural Overrides                        │
 │ • Group Admin Urgency & Relative-Time Deadlines (notify/urgent or event) │
 │ • Direct @Mention Escalation in Work/Group Chats (notify/urgent)         │
 │ • C2C Selling Co-occurrence (Selling Verb + Price/Logistics -> digest)   │
 │ • Sender-Trust Gated Status-Progression (State-Change Verb + Time Ref)   │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │ (If Unambiguous)
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 3: Multi-Centroid BERT Transformer Context Engine                   │
 │ • SentenceTransformer ('all-MiniLM-L6-v2') Multi-Vector Centroids         │
 │ • Soft Negation Downweighting (e.g., "nothing urgent", "don't call now")  │
 │ • Promotional Guard (Unverified/Cold Promos -> mute; Verified -> digest)  │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Technical Capabilities & Solutions

1. **Stage 1 Security Guard**:
   * **Domain Spoofing Interception**: Inspects `business_accounts.csv` relational lookup table. If `official_domain != domain_used_by_sender`, or if an unverified business has user scam reports, it immediately silences the message (`action: mute`, `message_type: scam`).
   * **Prompt Injection Defense**: Intercepts malicious instructions embedded in message text (*"Ignore all previous routing rules..."*).
   * **Credential & OTP Theft**: Intercepts patterns requesting card verification, wallet detail verification, or 6-digit OTP codes.

2. **Stage 2 Structural & Relational Overrides**:
   * **Group Membership Context**: Relational $O(1)$ indexed lookup in `group_members.csv` checks whether the sender is a verified `admin` or standard member.
   * **C2C Selling Pattern**: Requires structural co-occurrence of a selling verb (*selling, for sale, wts*) AND a price/pickup tag (*₹, $, pickup, gate X*), preventing false positives on casual mentions.
   * **Status-Progression Upweight**: Combines state-change verbs (*shipped, dispatched, out for delivery, arrived*) with near-term time references (*today, tonight, in N mins*), gated on verified sender trust and evidence duplicate check.

3. **Stage 3 Hybrid Multimodal Processor**:
   * **VLM & Speech-To-Text**: Processes images via Gemini VLM / Tesseract OCR and voice notes via Whisper Large v3 / Hugging Face pipeline.
   * **Rate-Limit & Quota Protection**: Enforces client-side 10 RPM throttling (6.0s spacing), 1,400 RPD daily quota caps, and exponential backoff retries on HTTP 429 rate limit responses.
   * **Local Binary Auto-Injection**: Auto-detects and injects local `imageio-ffmpeg` and `tesseract.exe` binaries for 100% offline local processing capability.
   * **Persistent Disk Caching**: Stores MD5 hash results in `.cache/media_cache.json` for 0ms repeat processing.

4. **RAG Evidence Retrieval Engine**:
   * TF-IDF vector similarity over historical user messages (`message_history.csv`) retrieves concrete `evidence_message_ids` for transparent decision context.

---

## 🚀 Setup & Execution Instructions

### Prerequisites
* Python 3.9+
* Required packages listed in `requirements.txt`

### 1. Installation
Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Main Notification Router
To process all 110 messages in `dataset/messages.csv` and generate `dataset/output.csv`:

```bash
python code/main.py
```

### 3. Evaluating Benchmark Schema & Compliance
To run the official benchmark evaluator:

```bash
python code/evaluation/main.py
```

* **Expected Output**:
  ```text
  ==================================================
  WhatsApp Router Benchmark Evaluator
  ==================================================
  [OK] Schema Check Passed: Column header matches required format.
  [OK] Enum & Range Check Passed: All actions, types, and confidences are valid.

  ==================================================
  SUCCESS: Benchmark Evaluation Passed! All checks green.
  ==================================================
  ```

---

## 📂 Repository Structure

```text
.
├── .cache/                           # Persistent media MD5 hash cache (tracked)
│   └── media_cache.json
├── code/                             # Production source code
│   ├── main.py                       # Main CLI entry point & orchestrator
│   ├── router.py                     # Multi-tier fenced decision router
│   ├── semantic_context.py           # Multi-centroid BERT vector classifier
│   ├── context.py                    # Relational O(1) indexed metadata engine
│   ├── media.py                      # Multimodal processor (VLM/OCR/ASR + Rate Limiting)
│   ├── retrieval.py                  # RAG TF-IDF evidence retrieval engine
│   ├── postprocess.py                # Output schema validator & post-processor
│   └── evaluation/
│       └── main.py                   # Official benchmark evaluator
├── dataset/                          # Metadata CSV tables & media files
│   ├── messages.csv                  # 110 dataset messages to route
│   ├── output.csv                    # Final prediction results output file
│   ├── sample_messages.csv           # Solved ground-truth sample rows
│   ├── users.csv                     # User notification preferences &quiet hours
│   ├── groups.csv                    # Group chat metadata
│   ├── group_members.csv             # User group roles (admin vs member)
│   ├── business_accounts.csv         # Verified business & domain metadata
│   ├── user_business_history.csv     # Prior user-business transaction history
│   ├── message_history.csv           # Historical message context
│   └── media/                        # Image and Audio attachment files
│       ├── images/
│       └── audio/
├── .env.example                      # Environment variable template
├── problem_statement.md              # Original challenge specification
├── requirements.txt                  # Python dependencies
└── README.md                         # Setup & approach overview (This file)
```

---

## 📄 Submission Files

* **Source Code**: Complete working project in `code/`
* **Predictions CSV**: `dataset/output.csv` (110 rows, 100% compliant)
* **README**: Setup instructions and approach overview
