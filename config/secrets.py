"""Fetch secrets from AWS Secrets Manager using the instance role."""
import json
import functools
import boto3

REGION = "us-east-1"


@functools.lru_cache(maxsize=1)
def get_db_credentials() -> dict:
    """
    Read the analytics/postgres secret. Cached for the process lifetime --
    credentials don't change mid-run, saving a call per worker boot.
    """
    client = boto3.client("secretsmanager", region_name=REGION)
    raw = client.get_secret_value(SecretId="analytics/postgres")
    secret = json.loads(raw["SecretString"])
    return {
        "username": secret["username"],
        "password": secret["password"],
        "port": str(secret.get("port", 5432)),
        "dbname": secret.get("dbname", "analytics"),
    }


@functools.lru_cache(maxsize=1)
def get_polygon_key() -> str:
    """Polygon API key, stored as plain text in analytics/polygon."""
    client = boto3.client("secretsmanager", region_name=REGION)
    raw = client.get_secret_value(SecretId="analytics/polygon")
    return raw["SecretString"]