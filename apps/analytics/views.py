"""
API views for analytics.

Deliberately thin. Each view validates a little, calls one service
function, and serializes the result. All the real work -- fetching,
computing, persisting, caching -- lives in services.py. If a view grows
past ~15 lines of logic, that logic belongs in the service instead.

The service returns an AnalysisRun whether it succeeded or failed, so the
views return 200 with a status field rather than raising on a failed
analysis. A failed run is a valid result the client should see ("no closed
trades yet"), not an HTTP error. Genuine 4xx/5xx are reserved for bad
requests and unexpected faults.
"""

from __future__ import annotations

from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response

from apps.analytics import services
from apps.analytics.serializers import AnalysisRunSerializer
from common.auth import CognitoAuthentication


def _force_flag(request: Request) -> bool:
    return request.query_params.get("force", "").lower() in ("1", "true", "yes")


@api_view(["GET"])
@authentication_classes([CognitoAuthentication])
@permission_classes([IsAuthenticated])
def risk_analysis(request: Request) -> Response:
    """
    GET /api/v1/analytics/risk[?force=true]

    Portfolio risk metrics for the authenticated user's current holdings.
    Served from cache unless force is set.
    """
    run = services.get_or_run_risk(
        request.user.app_user,
        force=_force_flag(request),
    )
    return Response(
        AnalysisRunSerializer(run).data,
        # A failed analysis is still a 200 with a status the client reads.
        status=status.HTTP_200_OK,
    )


@api_view(["GET"])
@authentication_classes([CognitoAuthentication])
@permission_classes([IsAuthenticated])
def behavior_analysis(request: Request) -> Response:
    """
    GET /api/v1/analytics/behavior[?force=true]

    Trading-behaviour metrics from the user's closed lots.
    """
    run = services.get_or_run_behavior(
        request.user.app_user,
        force=_force_flag(request),
    )
    return Response(AnalysisRunSerializer(run).data, status=status.HTTP_200_OK)


@api_view(["GET"])
@authentication_classes([CognitoAuthentication])
@permission_classes([IsAuthenticated])
def health(request: Request) -> Response:
    """Liveness probe for the load balancer / uptime checks. No auth data used."""
    return Response({"status": "ok"})