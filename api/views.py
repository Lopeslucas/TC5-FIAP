from rest_framework import viewsets
from .models import Applicant, Vaga
from .serializers import ApplicantSerializer, VagaSerializer

class ApplicantViewSet(viewsets.ReadOnlyModelViewSet):  # somente leitura
    queryset = Applicant.objects.all()
    serializer_class = ApplicantSerializer

class VagaViewSet(viewsets.ReadOnlyModelViewSet):  # somente leitura
    queryset = Vaga.objects.all()
    serializer_class = VagaSerializer