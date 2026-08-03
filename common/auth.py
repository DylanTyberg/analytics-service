"""
Cognito JWT authentication for DRF.

Validates the access/ID token minted by the same Cognito pool the rest of
the platform uses, so the analytics service trusts the front end's
existing login rather than running its own auth.

Local development can bypass this with a stub user -- see AUTH_STUB in
settings -- so the schema and API can be built without wiring real tokens.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

import requests
from django.conf import settings
from jose import jwt
from rest_framework import authentication, exceptions

from apps.trading.models import AppUser


class CognitoUser:
    """
    Lightweight authenticated principal.

    Wraps the AppUser row (creating it on first sight) plus the token
    claims. Not a Django auth User -- this service has no user model of its
    own, only the analytical shadow keyed on the Cognito sub.
    """

    def __init__(self, app_user: AppUser, claims: dict):
        self.app_user = app_user
        self.claims = claims
        self.is_authenticated = True

    @property
    def sub(self) -> str:
        return self.claims["sub"]


@lru_cache(maxsize=1)
def _jwks(region: str, pool_id: str) -> dict:
    """
    Fetch and cache the pool's public signing keys.

    Cached process-wide: the keys rotate rarely, and refetching per request
    would add a network round trip to every call.
    """
    url = (
        f"https://cognito-idp.{region}.amazonaws.com/"
        f"{pool_id}/.well-known/jwks.json"
    )
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json()


class CognitoAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        # Local dev escape hatch: skip real validation, act as a fixed user.
        if getattr(settings, "AUTH_STUB", False):
            app_user, _ = AppUser.objects.get_or_create(cognito_sub="local-dev-user")
            return (CognitoUser(app_user, {"sub": "local-dev-user"}), None)

        header = authentication.get_authorization_header(request).decode("utf-8")
        if not header.startswith("Bearer "):
            return None  # no credentials -> let other authenticators try / 401

        token = header.split(" ", 1)[1].strip()
        claims = self._verify(token)

        app_user, _ = AppUser.objects.get_or_create(cognito_sub=claims["sub"])
        return (CognitoUser(app_user, claims), None)

    def _verify(self, token: str) -> dict:
        region = settings.COGNITO_REGION
        pool_id = settings.COGNITO_USER_POOL_ID

        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.JWTError:
            raise exceptions.AuthenticationFailed("malformed token")

        keys = _jwks(region, pool_id)["keys"]
        key = next((k for k in keys if k["kid"] == unverified_header.get("kid")), None)
        if key is None:
            # kid not found -- keys may have rotated; drop the cache and retry once.
            _jwks.cache_clear()
            keys = _jwks(region, pool_id)["keys"]
            key = next((k for k in keys if k["kid"] == unverified_header.get("kid")), None)
            if key is None:
                raise exceptions.AuthenticationFailed("unknown signing key")

        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=settings.COGNITO_APP_CLIENT_ID,
                issuer=f"https://cognito-idp.{region}.amazonaws.com/{pool_id}",
            )
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("token expired")
        except jwt.JWTError as e:
            raise exceptions.AuthenticationFailed(f"invalid token: {e}")

        return claims

    def authenticate_header(self, request):
        # Makes DRF return 401 (not 403) when auth is missing.
        return "Bearer"