# urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import *
from django.contrib import admin
from api.views import test_metrics

# Router para curriculos e vagas
router = DefaultRouter()
router.register(r'curriculos', CurriculoViewSet)
router.register(r'vagas', VagaViewSet)

urlpatterns = [
    # Vagas e currículos
    path('', include(router.urls)),

    # Endpoints específicos de ML
    path('ml/load/', LoadArtifactsView.as_view(), name='ml-load'),
    path('ml/train/', TrainModelView.as_view(), name='ml-train'),
    path('ml/metrics/', MetricsView.as_view(), name='ml-metrics'),
    path('ml/health/', HealthView.as_view(), name='ml-health'),
    path('api/test-metrics/', test_metrics, name='test_metrics'),
]
