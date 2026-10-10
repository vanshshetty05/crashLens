"""WSGI entry point for hosts that start the app with ``gunicorn app:app``."""
from api.app import create_app

app = create_app()
