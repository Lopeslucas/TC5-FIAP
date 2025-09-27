# urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CurriculoViewSet, VagaViewSet
from django.contrib import admin

router = DefaultRouter()
router.register(r'curriculos', CurriculoViewSet)
router.register(r'vagas', VagaViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
