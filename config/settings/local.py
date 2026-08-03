from .base import *  # noqa

SECRET_KEY = "django-insecure-local-only-not-a-real-secret"
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "analytics"),
        "USER": os.environ.get("POSTGRES_USER", "analytics"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "localdev"),
        "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "PORT": os.environ.get("POSTGRES_PORT", "5434"),
    }
}
AUTH_STUB = True

CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]