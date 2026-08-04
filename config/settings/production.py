import os
from .base import *  # noqa
from config.secrets import get_db_credentials, get_polygon_key

DEBUG = False
SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]
ALLOWED_HOSTS = os.environ["DJANGO_ALLOWED_HOSTS"].split(",")

COGNITO_REGION = "us-east-1"
COGNITO_USER_POOL_ID = "us-east-1_9VbCwStHJ"       
COGNITO_APP_CLIENT_ID = "8trkun3hdfm03ngugqclo5q7f" 

_creds = get_db_credentials()
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": _creds["dbname"],
        "USER": _creds["username"],
        "PASSWORD": _creds["password"],
        "HOST": os.environ["RDS_HOST"],
        "PORT": _creds["port"],
        "CONN_MAX_AGE": 600,
    }
}

POLYGON_API_KEY = get_polygon_key()
USER_DATA_TABLE = os.environ.get("USER_DATA_TABLE", "stock-user-data")

# Behind nginx doing TLS termination.
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

CORS_ALLOWED_ORIGINS = [
    "https://fintech-platform.htytun.com",
    "http://localhost:3000",
]