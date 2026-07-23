from .base import *  # noqa
from config.secrets import get_db_credentials

DEBUG = False
SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]
ALLOWED_HOSTS = os.environ["DJANGO_ALLOWED_HOSTS"].split(",")

_creds = get_db_credentials()
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": _creds["dbname"],
        "USER": _creds["username"],
        "PASSWORD": _creds["password"],
        "HOST": _creds["host"],
        "PORT": _creds.get("port", "5432"),
        "CONN_MAX_AGE": 600,
    }
}

SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")