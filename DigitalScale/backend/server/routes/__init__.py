"""
Módulo de rutas y controladores HTTP de la API.
"""

from .health_routes import router as health_router
from .predict_routes import router as predict_router

__all__ = ["health_router", "predict_router"]
