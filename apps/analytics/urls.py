# apps/analytics/urls.py
from django.urls import path

from apps.analytics import views

urlpatterns = [
    path("analytics/risk", views.risk_analysis, name="risk-analysis"),
    path("analytics/behavior", views.behavior_analysis, name="behavior-analysis"),
    path("health", views.health, name="health"),
]

# ---------------------------------------------------------------------
# Wire into config/urls.py:
#
#   from django.urls import path, include
#   urlpatterns = [
#       path("api/v1/", include("apps.analytics.urls")),
#   ]
#
# And in config/settings/base.py add DRF + the default auth/permission:
#
#   INSTALLED_APPS += ["rest_framework"]
#   REST_FRAMEWORK = {
#       "DEFAULT_AUTHENTICATION_CLASSES": ["common.auth.CognitoAuthentication"],
#       "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
#       "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
#   }
#
# In config/settings/local.py, flip on the auth stub so you can hit the API
# without real Cognito tokens:
#
#   AUTH_STUB = True
#
# In production.py leave AUTH_STUB unset/False and set:
#   COGNITO_REGION, COGNITO_USER_POOL_ID, COGNITO_APP_CLIENT_ID
# ---------------------------------------------------------------------