from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock
import pandas as pd
import torch

from api.models import Curriculo, Vaga

class TestAPICase(APITestCase):
    
    def setUp(self):
        self.client = APIClient()
        # Criar dados básicos para Curriculo e Vaga
        self.curriculo = Curriculo.objects.create(
            cv_pt="Currículo teste", cv_sugerido="Texto sugerido"
        )
        self.vaga = Vaga.objects.create(
            titulo_vaga="Analista", 
            areas_atuacao="TI",
            principais_atividades="Atividades teste"
        )

    # ------------------------------
    # Endpoints GET simples
    # ------------------------------
    def test_health_endpoint(self):
        url = reverse('ml-health')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("status", response.data)
        self.assertIn("details", response.data)

    def test_list_curriculos(self):
        url = reverse('curriculo-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(len(response.data) >= 1)

    def test_list_vagas(self):
        url = reverse('vaga-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(len(response.data) >= 1)

    # ------------------------------
    # Endpoints que dependem de pipeline
    # ------------------------------
    @patch("api.views._pipeline")
    def test_load_artifacts_success(self, mock_pipeline):
        mock_pipeline.load_data.return_value = (
            pd.DataFrame([{"cv_sugerido": "Texto"}]),
            pd.DataFrame([{"titulo_vaga": "Vaga"}])
        )
        mock_pipeline.embedding_processor.get_cv_embeddings.return_value = torch.rand(1, 768)

        url = reverse('ml-load')
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("curriculos", response.data)
        self.assertIn("vagas", response.data)

    @patch("api.views._pipeline")
    def test_metrics_endpoint(self, mock_pipeline):
        mock_pipeline.get_metrics.return_value = {"accuracy": 0.95, "loss": 0.05}
        url = reverse('ml-metrics')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("accuracy", response.data)
        self.assertIn("loss", response.data)

    @patch("api.views._pipeline")
    def test_train_model_success(self, mock_pipeline):
        mock_pipeline.run_pipeline.return_value = pd.DataFrame([{"cv_id":1,"vaga_id":1}])
        mock_pipeline.train_model.return_value = (MagicMock(), MagicMock(), {"accuracy":0.9})

        url = reverse('ml-train')
        response = self.client.post(url, data={"epochs": 1, "batch_size": 1, "learning_rate": 0.001})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("metrics", response.data)

    # ------------------------------
    # Endpoint vaga-predict
    # ------------------------------
    @patch("api.views._pipeline")
    @patch("api.views._df_cvs", new_callable=lambda: pd.DataFrame([{"cv_id": 1, "cv_sugerido": "Texto"}]))
    @patch("api.views._df_vagas", new_callable=lambda: pd.DataFrame([{"id": 1, "titulo_vaga": "Analista", "principais_atividades": "Atividades"}]))
    @patch("api.views._cv_embeddings", new_callable=lambda: torch.rand(1, 768))
    def test_vaga_predict_success(self, mock_embeddings, mock_df_vagas, mock_df_cvs, mock_pipeline):
        # Mock do método que retorna candidatos
        mock_pipeline.embedding_processor.encode_text.return_value = torch.rand(1, 768)
        mock_pipeline.get_candidates.return_value = [{"cv_id": 1, "score": 0.95}]

        url = reverse('vaga-predict', kwargs={"pk": self.vaga.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("candidatos", response.data)