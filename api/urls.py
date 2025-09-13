from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ApplicantViewSet, VagaViewSet

router = DefaultRouter()
router.register(r'applicants', ApplicantViewSet)
router.register(r'vagas', VagaViewSet)

urlpatterns = [
    path('', include(router.urls)),
]