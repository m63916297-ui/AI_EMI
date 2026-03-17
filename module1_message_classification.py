"""
Module 1: Message Classification Using an LLM
==============================================
A robust classification system for multi-channel business communications
using Large Language Models with structured outputs and comprehensive error handling.

Categories: SUPPORT, SALES, QUOTATION, COMPLAINT_ESCALATION, SUPPLIER, SPAM_IRRELEVANT
"""

import json
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MessageCategory(Enum):
    SUPPORT = "SUPPORT"
    SALES = "SALES"
    QUOTATION = "QUOTATION"
    COMPLAINT_ESCALATION = "COMPLAINT_ESCALATION"
    SUPPLIER = "SUPPLIER"
    SPAM_IRRELEVANT = "SPAM_IRRELEVANT"


@dataclass
class ClassificationResult:
    category: MessageCategory
    confidence: float
    reasoning: str
    alternative_category: Optional[MessageCategory]
    needs_human_review: bool


SYSTEM_PROMPT = """You are a message classification assistant for a business organization.
Your task is to classify incoming messages into exactly ONE category.

=== CATEGORY DEFINITIONS ===

1. SUPPORT
   - Technical issues, bugs, or product malfunctions
   - Questions about how to use a product or service
   - Requests for help with account access or login problems
   - Troubleshooting inquiries

2. SALES
   - Inquiries about products, services, or pricing
   - Requests for product demonstrations or meetings
   - Questions about availability or features before purchase
   - Lead generation or qualification inquiries

3. QUOTATION
   - Explicit requests for price quotes or formal proposals
   - Questions about bulk pricing or volume discounts
   - Requests for written cost estimates
   - Tender or bid-related communications

4. COMPLAINT_ESCALATION
   - Expressions of dissatisfaction or frustration
   - Threats to escalate to management or legal
   - Reports of service failures or poor experiences
   - Requests to speak with supervisors or managers

5. SUPPLIER
   - Communications from vendors or business partners
   - Invoice or payment-related messages from suppliers
   - Procurement or supply chain inquiries
   - B2B partnership proposals

6. SPAM_IRRELEVANT
   - Unsolicited marketing or promotional content
   - Messages with no clear business purpose
   - Empty, garbled, or nonsensical content
   - Phishing attempts or suspicious links

=== OUTPUT FORMAT ===

You MUST respond with a valid JSON object in this exact format:
{
  "category": "CATEGORY_NAME",
  "confidence": 0.95,
  "reasoning": "Brief explanation of classification decision",
  "alternative_category": "SECOND_BEST_MATCH or null"
}

Constraints:
- category MUST be one of: SUPPORT, SALES, QUOTATION, COMPLAINT_ESCALATION, SUPPLIER, SPAM_IRRELEVANT
- confidence MUST be a decimal between 0.0 and 1.0
- reasoning MUST be 1-2 sentences maximum
- Output ONLY the JSON object, no additional text
"""


def build_classification_prompt(message: str, channel: str) -> str:
    """
    Constructs the complete prompt for the LLM.
    Separates system instructions from user input.
    """
    user_prompt = f"""
=== MESSAGE TO CLASSIFY ===
Channel: {channel}
Content: {message}

Please classify this message according to the defined categories.
"""
    return SYSTEM_PROMPT + "\n" + user_prompt


def simulate_llm_response(prompt: str, message: str) -> str:
    """
    Simulates LLM API response for demonstration.
    In production, this would call OpenAI, Anthropic, or similar API.
    """
    message_lower = message.lower()

    if any(
        word in message_lower
        for word in ["quote", "price", "cost", "estimate", "pricing", "quotation"]
    ):
        return json.dumps(
            {
                "category": "QUOTATION",
                "confidence": 0.92,
                "reasoning": "Message explicitly requests pricing information.",
                "alternative_category": "SALES",
            }
        )

    elif any(
        word in message_lower
        for word in [
            "problem",
            "issue",
            "bug",
            "error",
            "not working",
            "can't access",
            "login",
        ]
    ):
        return json.dumps(
            {
                "category": "SUPPORT",
                "confidence": 0.88,
                "reasoning": "Reports a technical issue requiring assistance.",
                "alternative_category": "COMPLAINT_ESCALATION",
            }
        )

    elif any(
        word in message_lower
        for word in [
            "unacceptable",
            "frustrated",
            "complaint",
            "escalate",
            "manager",
            "supervisor",
            "terrible",
            "worst",
        ]
    ):
        return json.dumps(
            {
                "category": "COMPLAINT_ESCALATION",
                "confidence": 0.87,
                "reasoning": "Expression of dissatisfaction with intent to escalate.",
                "alternative_category": "SUPPORT",
            }
        )

    elif any(
        word in message_lower
        for word in [
            "vendor",
            "supplier",
            "invoice",
            "payment",
            "procurement",
            "b2b",
            "partnership",
        ]
    ):
        return json.dumps(
            {
                "category": "SUPPLIER",
                "confidence": 0.90,
                "reasoning": "Communication from vendor or business partner.",
                "alternative_category": None,
            }
        )

    elif any(
        word in message_lower
        for word in [
            "free",
            "winner",
            "congratulations",
            "click here",
            "limited time",
            "act now",
            "!!!",
            "bit.ly",
        ]
    ):
        return json.dumps(
            {
                "category": "SPAM_IRRELEVANT",
                "confidence": 0.99,
                "reasoning": "Classic spam pattern detected.",
                "alternative_category": None,
            }
        )

    elif any(
        word in message_lower
        for word in [
            "product",
            "service",
            "demo",
            "meeting",
            "features",
            "availability",
            "interested",
        ]
    ):
        return json.dumps(
            {
                "category": "SALES",
                "confidence": 0.85,
                "reasoning": "General product or service inquiry.",
                "alternative_category": "QUOTATION",
            }
        )

    else:
        return json.dumps(
            {
                "category": "SPAM_IRRELEVANT",
                "confidence": 0.50,
                "reasoning": "No clear business purpose detected.",
                "alternative_category": "SALES",
            }
        )


def parse_llm_output(raw_output: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parses LLM output with robust error handling.
    Returns (parsed_dict, error_message).
    """
    try:
        json_match = re.search(r"\{[^{}]*\}", raw_output, re.DOTALL)
        if not json_match:
            return None, "No valid JSON object found in response"

        parsed = json.loads(json_match.group())

        required_fields = ["category", "confidence", "reasoning"]
        for field in required_fields:
            if field not in parsed:
                return None, f"Missing required field: {field}"

        return parsed, None

    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"


def classify_message(
    message: str,
    channel: str = "unknown",
    confidence_threshold: float = 0.70,
    require_human_review_below: float = 0.60,
) -> ClassificationResult:
    """
    Main classification function with comprehensive error handling.

    Args:
        message: The text content to classify
        channel: Source channel (email, whatsapp, web_form)
        confidence_threshold: Minimum confidence for auto-classification
        require_human_review_below: Confidence below this flags for review

    Returns:
        ClassificationResult with category, confidence, and review flags
    """

    prompt = build_classification_prompt(message, channel)

    raw_output = simulate_llm_response(prompt, message)

    parsed, error = parse_llm_output(raw_output)

    if error:
        return ClassificationResult(
            category=MessageCategory.SPAM_IRRELEVANT,
            confidence=0.0,
            reasoning=f"Classification error: {error}",
            alternative_category=None,
            needs_human_review=True,
        )

    try:
        category = MessageCategory[parsed["category"]]
    except KeyError:
        return ClassificationResult(
            category=MessageCategory.SPAM_IRRELEVANT,
            confidence=0.0,
            reasoning=f"Invalid category returned: {parsed['category']}",
            alternative_category=None,
            needs_human_review=True,
        )

    confidence = float(parsed["confidence"])
    needs_review = confidence < require_human_review_below

    alt_cat = None
    if parsed.get("alternative_category"):
        try:
            alt_cat = MessageCategory[parsed["alternative_category"]]
        except KeyError:
            pass

    return ClassificationResult(
        category=category,
        confidence=confidence,
        reasoning=parsed["reasoning"],
        alternative_category=alt_cat,
        needs_human_review=needs_review,
    )


def run_examples():
    """Run example classifications from Task 3."""

    print("=" * 70)
    print("MODULE 1 - TASK 3: EXAMPLE CLASSIFICATIONS")
    print("=" * 70)

    test_cases = [
        {
            "message": "I've been trying to log into my account for three days and it keeps saying 'invalid credentials' even though I'm 100% sure my password is correct. This is unacceptable - I need to access my reports for a client meeting tomorrow!",
            "channel": "email",
        },
        {
            "message": "Hi! Saw your product at the trade show. Can you send me pricing for the enterprise plan? We're a team of 50 looking to onboard next quarter.",
            "channel": "whatsapp",
        },
        {
            "message": "CONGRATULATIONS! You've been selected for a FREE iPhone 15! Click here to claim: bit.ly/free-iphone-2025 Limited time offer!!!",
            "channel": "web_form",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Example {i} ---")
        print(f"Channel: {test['channel']}")
        print(f"Message: {test['message'][:80]}...")

        result = classify_message(test["message"], test["channel"])

        print(f"Category: {result.category.value}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Needs Human Review: {result.needs_human_review}")
        if result.alternative_category:
            print(f"Alternative Category: {result.alternative_category.value}")


def detect_performance_degradation(metrics_history: list) -> Dict:
    """
    Detects performance degradation in the classifier based on historical metrics.

    Args:
        metrics_history: List of dictionaries with 'avg_confidence', 'review_rate',
                        'low_confidence_rate', 'accuracy' keys

    Returns:
        Dictionary with degradation status and alerts
    """
    if len(metrics_history) < 7:
        return {"status": "insufficient_data", "alerts": []}

    recent = metrics_history[-7:]
    baseline = (
        metrics_history[:-7] if len(metrics_history) > 14 else metrics_history[:7]
    )

    alerts = []

    avg_confidence_recent = sum(m.get("avg_confidence", 0) for m in recent) / len(
        recent
    )
    avg_confidence_baseline = sum(m.get("avg_confidence", 0) for m in baseline) / len(
        baseline
    )

    if (
        avg_confidence_baseline > 0
        and avg_confidence_recent < avg_confidence_baseline * 0.9
    ):
        alerts.append(
            {
                "type": "confidence_drop",
                "message": f"Average confidence dropped by {((1 - avg_confidence_recent / avg_confidence_baseline) * 100):.1f}%",
                "severity": "high",
            }
        )

    review_rate_recent = sum(m.get("review_rate", 0) for m in recent) / len(recent)
    if review_rate_recent > 0.2:
        alerts.append(
            {
                "type": "high_review_rate",
                "message": f"Human review rate increased to {review_rate_recent * 100:.1f}%",
                "severity": "medium",
            }
        )

    low_conf_rate_recent = sum(m.get("low_confidence_rate", 0) for m in recent) / len(
        recent
    )
    if low_conf_rate_recent > 0.3:
        alerts.append(
            {
                "type": "high_low_confidence",
                "message": f"Low confidence predictions at {low_conf_rate_recent * 100:.1f}%",
                "severity": "medium",
            }
        )

    return {
        "status": "degradation_detected" if alerts else "healthy",
        "alerts": alerts,
        "metrics": {
            "recent_avg_confidence": avg_confidence_recent,
            "baseline_avg_confidence": avg_confidence_baseline,
            "recent_review_rate": review_rate_recent,
            "recent_low_confidence_rate": low_conf_rate_recent,
        },
    }


if __name__ == "__main__":
    run_examples()

    print("\n" + "=" * 70)
    print("PERFORMANCE DEGRADATION DETECTION DEMO")
    print("=" * 70)

    sample_metrics = [
        {
            "avg_confidence": 0.85,
            "review_rate": 0.05,
            "low_confidence_rate": 0.10,
            "accuracy": 0.92,
        },
        {
            "avg_confidence": 0.84,
            "review_rate": 0.06,
            "low_confidence_rate": 0.11,
            "accuracy": 0.91,
        },
        {
            "avg_confidence": 0.86,
            "review_rate": 0.05,
            "low_confidence_rate": 0.09,
            "accuracy": 0.93,
        },
        {
            "avg_confidence": 0.83,
            "review_rate": 0.08,
            "low_confidence_rate": 0.14,
            "accuracy": 0.89,
        },
        {
            "avg_confidence": 0.75,
            "review_rate": 0.15,
            "low_confidence_rate": 0.25,
            "accuracy": 0.82,
        },
        {
            "avg_confidence": 0.72,
            "review_rate": 0.18,
            "low_confidence_rate": 0.28,
            "accuracy": 0.80,
        },
        {
            "avg_confidence": 0.70,
            "review_rate": 0.22,
            "low_confidence_rate": 0.32,
            "accuracy": 0.78,
        },
    ]

    result = detect_performance_degradation(sample_metrics)
    print(f"Status: {result['status']}")
    for alert in result["alerts"]:
        print(f"  - [{alert['severity']}] {alert['message']}")
