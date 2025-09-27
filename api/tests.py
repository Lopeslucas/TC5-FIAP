from django.test import TestCase
from rest_framework.test import APITestCase
from django.urls import reverse

from api.models import Applicant, Vaga


# -------------------------------
# Testes de Models
# -------------------------------
class ApplicantModelTest(TestCase):
    def setUp(self):
        self.applicant = Applicant.objects.create(
            nome="Lucas Lopes",
            email="lucas@example.com",
            local="São Paulo"
        )

    def test_applicant_created(self):
        """Verifica se o Applicant foi criado corretamente"""
        self.assertEqual(Applicant.objects.count(), 1)
        self.assertEqual(self.applicant.nome, "Lucas Lopes")
        self.assertEqual(self.applicant.email, "lucas@example.com")


class VagaModelTest(TestCase):
    def setUp(self):
        self.vaga = Vaga.objects.create(
            titulo="Desenvolvedor Python",
            descricao="Desenvolver APIs com Django",
            local="Remoto"
        )

    def test_vaga_created(self):
        """Verifica se a Vaga foi criada corretamente"""
        self.assertEqual(Vaga.objects.count(), 1)
        self.assertEqual(self.vaga.titulo, "Desenvolvedor Python")
        self.assertEqual(self.vaga.local, "Remoto")


# -------------------------------
# Testes da API
# -------------------------------
class ApplicantAPITest(APITestCase):
    def setUp(self):
        self.applicant = Applicant.objects.create(
            nome="Maria Silva",
            email="maria@example.com",
            local="Rio de Janeiro"
        )

    def test_list_applicants(self):
        """Testa se a API retorna lista de Applicants"""
        url = reverse("applicant-list")  # precisa estar no router do DRF
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["nome"], "Maria Silva")


class VagaAPITest(APITestCase):
    def setUp(self):
        self.vaga = Vaga.objects.create(
            titulo="Analista de Dados",
            descricao="Criar dashboards de BI",
            local="São Paulo"
        )

    def test_list_vagas(self):
        """Testa se a API retorna lista de Vagas"""
        url = reverse("vaga-list")  # precisa estar no router do DRF
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["titulo"], "Analista de Dados")