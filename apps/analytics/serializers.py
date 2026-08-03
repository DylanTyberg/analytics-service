"""
Serializers for analysis results.

These shape the persisted AnalysisRun + metrics rows into the JSON the
front end consumes. Read-only throughout -- results are produced by the
service layer, never created via the API.
"""

from __future__ import annotations

from rest_framework import serializers

from apps.analytics.models import AnalysisRun, BehaviorMetrics, RiskMetrics


class RiskMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = RiskMetrics
        exclude = ["id", "run"]


class BehaviorMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = BehaviorMetrics
        exclude = ["id", "run"]


class AnalysisRunSerializer(serializers.ModelSerializer):
    """
    The run envelope plus whichever metrics block it produced. The metrics
    are nested so the client gets status and results in one response.
    """

    metrics = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisRun
        fields = [
            "id", "run_type", "status", "as_of", "lookback_days",
            "securities_included", "securities_excluded",
            "error_message", "created_at", "completed_at", "duration_ms",
            "metrics",
        ]

    def get_metrics(self, run: AnalysisRun):
        if run.status != AnalysisRun.Status.SUCCEEDED:
            return None
        if run.run_type == AnalysisRun.RunType.RISK:
            block = getattr(run, "risk_metrics", None)
            return RiskMetricsSerializer(block).data if block else None
        if run.run_type == AnalysisRun.RunType.BEHAVIOR:
            block = getattr(run, "behavior_metrics", None)
            return BehaviorMetricsSerializer(block).data if block else None
        return None