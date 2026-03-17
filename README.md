# AI Systems Assessment - Technical Implementation

This project contains the complete implementation for the AI Systems Assessment, covering message classification, retrieval-augmented generation, and anomaly detection system design.

---

## Module 1: Message Classification Using an LLM

### Overview
A robust classification system for multi-channel business communications using Large Language Models with structured outputs and comprehensive error handling.

### Tasks

**Task 1 - Prompt Design**
- Designed a system prompt with 6 categories: SUPPORT, SALES, QUOTATION, COMPLAINT_ESCALATION, SUPPLIER, SPAM_IRRELEVANT
- Structured JSON output format with confidence scores and alternative categories

**Task 2 - Classification Function**
- Implemented `classify_message()` function with:
  - Prompt building
  - Simulated LLM response (keyword-based)
  - Robust JSON parsing with error handling
  - Human review flagging based on confidence thresholds

**Task 3 - Examples**
- Three example messages demonstrating classification:
  1. Complaint/Escalation: Login issues with frustration
  2. Quotation: Pricing request for enterprise plan
  3. Spam: Fake prize notification

**Additional: Performance Degradation Detection**
- Monitoring strategy using:
  - Confidence distribution analysis
  - Human review rate tracking
  - Low confidence prediction alerts

---

## Module 2: Mini RAG System

### Overview
A Retrieval-Augmented Generation system for internal operational question answering with source citation and hallucination prevention.

### Tasks

**Task 1 - Implementation**
- Knowledge base with 3 documents:
  - DOC_1: Support hours (Monday-Friday, 8am-6pm)
  - DOC_2: Quotation response time (24 business hours)
  - DOC_3: Contract approval requirement

- Multi-signal relevance scoring:
  - Keyword overlap (30%)
  - Semantic expansion matching (20%)
  - Question-type detection (40%)

**Task 2 - Explanation**
- Document relevance determination methodology
- Hallucination prevention strategies:
  - Answer generation constraint
  - Explicit fallback messages
  - Source attribution
  - No extrapolation

**Additional: Scaling to 100,000 Documents**
- Vector embeddings approach
- Chunking strategy
- Hybrid search with reranking
- Cost management techniques

---

## Module 3: Technical Design Review - Anomaly Detection System

### Overview
Critical evaluation and improvement of an LLM-generated system design for production deployment.

### Tasks

**Task 1 - Structured Technical Review**
- Identified 6 critical concerns:
  - Data quality assumptions
  - Unspecified algorithm choice
  - Misleading accuracy metrics
  - Vague "real-time" requirements
  - Missing drift detection
  - Absence of governance

**Task 2 - Improved System Design**
- Hybrid approach: SPC + ML + Domain rules
- Critical data characteristics
- Mandatory human review points

**Task 3 - Evaluation & Validation Strategy**
- Metrics to use: Precision@k, Recall, MTTD, FPR
- Metrics to avoid: Overall accuracy, F1 without context
- Pre-production validation: Backtesting, Shadow mode, Stress testing
- Post-deployment monitoring

**Task 4 - Corrected Technical Summary**
- Engineering-ready system description with concrete specifications

---

## File Structure

```
AI-EMI1/
├── module1_message_classification.py  # Message classification system
├── module2_mini_rag.py               # RAG system implementation
├── module3_anomaly_detection.py       # Anomaly detection review
├── README.md                          # This file
└── task.txt                           # Original task specification
```

---

## Running the Code

```bash
# Module 1 - Message Classification
python module1_message_classification.py

# Module 2 - Mini RAG System
python module2_mini_rag.py

# Module 3 - Anomaly Detection
python module3_anomaly_detection.py
```

---

## Key Features

- **Type Safety**: Enums and dataclasses for robust type handling
- **Error Handling**: Comprehensive try-catch blocks with fallback mechanisms
- **Structured Outputs**: JSON format for machine-readable results
- **Human-in-the-Loop**: Review flags for low-confidence predictions
- **Source Attribution**: Document citation for verification
- **Production-Ready**: Configurable thresholds and monitoring hooks
