# serializers.py

from rest_framework import serializers
from .models import Curriculo, Vaga

class CurriculoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Curriculo
        fields = "__all__"

class VagaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vaga
        fields = "__all__"
