"""
Module 3: Technical Design Review - Anomaly Detection System
==============================================================
Critical evaluation and improvement of an LLM-generated system design
for production deployment, addressing real-world operational concerns.

Initial Draft (LLM-Generated):
"The anomaly detection system uses machine learning models to monitor
operational data in real time. It learns normal behavior patterns and
automatically detects anomalies with high accuracy and minimal false
positives. The system is scalable, requires little human intervention,
and can be deployed across multiple operational domains."
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import random


class AnomalyType(Enum):
    POINT = "point_anomaly"
    CONTEXTUAL = "contextual_anomaly"
    COLLECTIVE = "collective_anomaly"


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataCharacteristic:
    name: str
    description: str
    requirement: str


@dataclass
class HumanReviewPoint:
    scenario: str
    reason: str


@dataclass
class EvaluationMetric:
    name: str
    purpose: str
    target: str


@dataclass
class AnomalyAlert:
    timestamp: datetime
    severity: AlertSeverity
    description: str
    is_true_positive: Optional[bool] = None


def task1_structured_technical_review():
    """
    Task 1: Structured Technical Review
    Identifies critical concerns with the initial LLM-generated design.
    """

    print("=" * 70)
    print("MODULE 3 - TASK 1: STRUCTURED TECHNICAL REVIEW")
    print("=" * 70)

    concerns = [
        {
            "category": "Data Assumptions and Stability",
            "concern": "Undefined data quality requirements",
            "details": "The proposal assumes data is available, clean, and representative. In production, operational data suffers from: missing values, sensor failures, delayed ingestion, format inconsistencies, and seasonal variations. A system built on unstable data will produce unreliable anomaly scores. Without data quality monitoring, the model may learn from corrupted data, leading to systematic detection failures.",
        },
        {
            "category": "Modeling Approach",
            "concern": "Algorithm choice is unspecified",
            "details": "'Machine learning models' is too vague. Different anomaly types require different approaches: point anomalies vs. contextual anomalies vs. collective anomalies. Unsupervised approaches (isolation forests, autoencoders) have different failure modes than supervised models. The claim of 'high accuracy' assumes the model type is appropriate for the anomaly patterns—which cannot be known without domain analysis.",
        },
        {
            "category": "Evaluation and Metrics",
            "concern": "'High accuracy' is meaningless for anomaly detection",
            "details": "Anomaly detection is inherently imbalanced—normal behavior dominates. A model predicting 'normal' for everything achieves 99%+ accuracy while detecting zero anomalies. Real metrics matter: precision, recall, F1, and business-specific costs of false positives vs. false negatives. Without defined evaluation criteria, the system cannot be validated or improved.",
        },
        {
            "category": "Production Deployment Realism",
            "concern": "'Real-time' without latency specifications",
            "details": "'Real-time' is not a technical specification. Is the requirement 100ms, 1 second, or 1 minute latency? What happens during traffic spikes? How does the system behave when downstream services are slow? Without concrete SLAs and backpressure handling, the system will fail under load. Additionally, alert volume at scale can overwhelm operations teams.",
        },
        {
            "category": "Monitoring and Long-term Maintenance",
            "concern": "No mention of concept drift or retraining",
            "details": "Operational patterns change over time—seasonal shifts, business growth, new product launches, process changes. A model trained on historical data will gradually degrade. The proposal lacks: drift detection mechanisms, retraining triggers, model versioning, A/B testing for new models, and rollback procedures. 'Little human intervention' ignores the essential human role in model maintenance.",
        },
        {
            "category": "Human Oversight and Operational Risk",
            "concern": "Absence of governance and escalation paths",
            "details": "Anomaly detection influences business decisions. Who reviews alerts? How are true anomalies distinguished from false positives? What happens when the system misses a critical event? Without defined review loops, accountability structures, and escalation procedures, the system creates operational risk rather than reducing it. The claim of 'minimal false positives' is especially dangerous—it creates complacency.",
        },
    ]

    for i, item in enumerate(concerns, 1):
        print(f"\n{i}. {item['category']}")
        print(f"   Concern: {item['concern']}")
        print(f"   Details: {item['details'][:200]}...")


def task2_improved_system_design():
    """
    Task 2: Improved System Design
    Provides a recommended approach with critical data characteristics.
    """

    print("\n" + "=" * 70)
    print("MODULE 3 - TASK 2: IMPROVED SYSTEM DESIGN")
    print("=" * 70)

    print("\n--- Recommended Anomaly Detection Approach ---")
    print("""
Hybrid approach combining statistical methods with machine learning:

1. Statistical baseline: Use established statistical process control (SPC) 
   methods like CUSUM or EWMA for detecting mean shifts and trend changes. 
   These are interpretable, well-understood, and require minimal training data.

2. Unsupervised ML layer: Apply isolation forest or one-class SVM for 
   detecting novel anomalies that don't fit known patterns. These work well 
   when labeled anomaly data is scarce.

3. Domain-specific rules: Encode business knowledge as hard rules 
   (e.g., 'transaction value > 3x historical max' = alert). These provide 
   guaranteed detection for known risk patterns.

Rationale: Statistical methods are robust and interpretable. Unsupervised ML 
captures unknown patterns. Rules ensure critical scenarios are never missed. 
The combination provides defense-in-depth.
""")

    print("\n--- Critical Data Characteristics ---")
    characteristics = [
        DataCharacteristic(
            name="Temporal consistency",
            description="Timestamps must be accurate and consistent",
            requirement="Time zones and daylight savings handled correctly",
        ),
        DataCharacteristic(
            name="Sufficient historical baseline",
            description="At least 2-3 full seasonal cycles",
            requirement="To establish normal patterns",
        ),
        DataCharacteristic(
            name="Known change points",
            description="Documentation of system changes",
            requirement="Deployments and business events that affect baseline",
        ),
        DataCharacteristic(
            name="Clear definition of 'normal'",
            description="Explicitly document expected behavior",
            requirement="Per operational domain",
        ),
        DataCharacteristic(
            name="Labeled incident history",
            description="Past anomalies with outcomes",
            requirement="For validation and supervised fine-tuning",
        ),
    ]

    for char in characteristics:
        print(f"\n- {char.name}")
        print(f"  Description: {char.description}")
        print(f"  Requirement: {char.requirement}")

    print("\n\n--- Mandatory Human Review Points ---")
    review_points = [
        HumanReviewPoint(
            scenario="Novel anomaly patterns",
            reason="System has no precedent; human judgment needed to classify and respond appropriately",
        ),
        HumanReviewPoint(
            scenario="High-impact alerts",
            reason="Anomalies affecting critical operations or significant revenue require human confirmation before action",
        ),
        HumanReviewPoint(
            scenario="Model confidence below threshold",
            reason="Uncertain predictions should not trigger automated responses",
        ),
        HumanReviewPoint(
            scenario="Alerts coinciding with known events",
            reason="Correlate with deployments, maintenance windows, or planned business changes to avoid false alarms",
        ),
        HumanReviewPoint(
            scenario="Pattern changes in monitored data",
            reason="Drift detection triggers require human assessment before retraining",
        ),
    ]

    for point in review_points:
        print(f"\n- {point.scenario}")
        print(f"  Reason: {point.reason}")


def task3_evaluation_validation():
    """
    Task 3: Evaluation & Validation Strategy
    Defines metrics to use, metrics to avoid, and validation approaches.
    """

    print("\n" + "=" * 70)
    print("MODULE 3 - TASK 3: EVALUATION & VALIDATION STRATEGY")
    print("=" * 70)

    print("\n--- Metrics to Use ---")
    metrics = [
        EvaluationMetric(
            name="Precision@k",
            purpose="Of top k alerts, how many are true anomalies",
            target="> 80% for top 10 daily alerts",
        ),
        EvaluationMetric(
            name="Recall on known incidents",
            purpose="Detection rate for labeled historical anomalies",
            target="> 95% for critical incidents",
        ),
        EvaluationMetric(
            name="Mean Time to Detection (MTTD)",
            purpose="Average delay between anomaly start and alert",
            target="Domain-specific SLA",
        ),
        EvaluationMetric(
            name="False Positive Rate per day",
            purpose="Alerts requiring human review that are not anomalies",
            target="< 5 per day (manageable volume)",
        ),
        EvaluationMetric(
            name="Alert Fatigue Index",
            purpose="Ratio of dismissed/ignored alerts",
            target="< 10% dismissal rate",
        ),
    ]

    print(f"\n{'Metric':<30} {'Purpose':<50} {'Target':<30}")
    print("-" * 110)
    for m in metrics:
        print(f"{m.name:<30} {m.purpose:<50} {m.target:<30}")

    print("\n\n--- Metrics to Avoid ---")
    avoid = [
        "Overall accuracy: Meaningless due to class imbalance",
        "F1 score without context: Treats false positives and false negatives equally, which is rarely appropriate",
        "AUC-ROC for extreme imbalance: Can be misleadingly high; use AUC-PR instead",
    ]

    for item in avoid:
        print(f"  - {item}")

    print("\n\n--- Pre-Production Validation Strategy ---")
    strategies = [
        (
            "Historical Backtesting",
            "Run the system on past data with known incidents. Measure detection rate and false positive volume.",
        ),
        (
            "Shadow Mode Deployment",
            "Run the system in production without generating alerts. Compare its outputs against actual incidents and operator judgments.",
        ),
        (
            "Stress Testing",
            "Inject synthetic anomalies into test data streams to verify detection under various conditions.",
        ),
        (
            "Red Team Exercises",
            "Have domain experts attempt to create anomalies that evade detection.",
        ),
        (
            "Threshold Calibration",
            "Tune alert thresholds using labeled validation data to achieve desired precision/recall tradeoff.",
        ),
    ]

    for name, desc in strategies:
        print(f"\n- {name}")
        print(f"  {desc}")

    print("\n\n--- Post-Deployment Monitoring Approach ---")
    monitoring = [
        (
            "Alert Outcome Tracking",
            "Log every alert with human disposition (true positive, false positive, unknown). Build a feedback dataset.",
        ),
        (
            "Daily/Weekly Metrics Dashboard",
            "Track precision, alert volume, and MTTD trends over time.",
        ),
        (
            "Drift Detection",
            "Monitor input data distributions and model score distributions for shifts.",
        ),
        (
            "Regular Model Audits",
            "Quarterly review of model performance with retraining decision criteria.",
        ),
        (
            "Incident Post-Mortems",
            "For missed anomalies or major false positive events, conduct root cause analysis and update the system accordingly.",
        ),
    ]

    for name, desc in monitoring:
        print(f"\n- {name}")
        print(f"  {desc}")


def task4_corrected_technical_summary():
    """
    Task 4: Corrected Technical Summary
    Provides an engineering-ready system description.
    """

    print("\n" + "=" * 70)
    print("MODULE 3 - TASK 4: CORRECTED TECHNICAL SUMMARY")
    print("=" * 70)

    summary = """
Revised System Description (Engineering-Ready):

The anomaly detection system combines statistical process control with machine 
learning models to monitor operational data, targeting a false positive rate 
of <5 alerts/day and >95% recall on known incident types. Detection operates 
with configurable latency (default: 5-minute aggregation windows). The system 
requires an initial 90-day historical baseline and assumes stable data 
ingestion with <1% missing values. Human review is mandatory for novel 
patterns, high-impact alerts, and low-confidence predictions. Continuous 
monitoring tracks precision, recall, and data drift, with model retraining 
triggered by sustained performance degradation. Initial deployment covers one 
operational domain, with expansion contingent on demonstrated reliability over 
a 6-month evaluation period.
"""
    print(summary)


def simulate_anomaly_detection():
    """
    Demonstrates a simplified anomaly detection simulation.
    """

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION SIMULATION")
    print("=" * 70)

    np.random.seed(42)

    normal_data = np.random.normal(loc=100, scale=10, size=1000)

    anomalies = [180, 195, 45, 30, 170]

    all_data = np.concatenate([normal_data, np.array(anomalies)])

    mean = np.mean(normal_data)
    std = np.std(normal_data)
    threshold = mean + 3 * std

    print(f"\nBaseline Statistics:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Std Dev: {std:.2f}")
    print(f"  Anomaly Threshold (3σ): {threshold:.2f}")

    print(f"\nDetection Results:")
    alerts = []
    for i, value in enumerate(all_data[-5:]):
        is_anomaly = value > threshold
        if is_anomaly:
            alerts.append(value)
            severity = (
                AlertSeverity.CRITICAL if value > mean + 4 * std else AlertSeverity.HIGH
            )
            print(
                f"  Index {i}: Value={value:.2f} -> ANOMALY DETECTED (Severity: {severity.value})"
            )
        else:
            print(f"  Index {i}: Value={value:.2f} -> Normal")

    print(f"\nDetection Summary:")
    print(f"  Total anomalies injected: {len(anomalies)}")
    print(f"  Anomalies detected: {len(alerts)}")
    print(f"  Detection rate: {len(alerts) / len(anomalies) * 100:.1f}%")
    print(f"  False positives: {max(0, len(alerts) - len(anomalies))}")


if __name__ == "__main__":
    task1_structured_technical_review()
    task2_improved_system_design()
    task3_evaluation_validation()
    task4_corrected_technical_summary()
    simulate_anomaly_detection()
