# views.py - VERSÃO CORRIGIDA

from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import viewsets
from .models import Curriculo, Vaga
from .serializers import CurriculoSerializer, VagaSerializer

# importar pipeline e carregar embeddings/modelo em memória
from ML.ML_bert import JobMatchingPipeline
import numpy as np
from sentence_transformers import util
import torch

# Inicializar pipeline
pipeline = JobMatchingPipeline("bucket-tc5")
df_cvs, df_vagas = pipeline.load_data()
cv_embeddings = pipeline.embedding_processor.get_cv_embeddings(
    df_cvs["cv_sugerido"].astype(str).tolist()
)

class CurriculoViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Curriculo.objects.all()
    serializer_class = CurriculoSerializer

class VagaViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Vaga.objects.all()
    serializer_class = VagaSerializer

    @action(detail=True, methods=["get"])
    def recomendacoes(self, request, pk=None):
        vaga = self.get_object()
        vaga_text = f"{vaga.titulo_vaga} {vaga.principais_atividades}"
        vaga_embedding = pipeline.embedding_processor.encode_text(vaga_text)

        # Converter ambos para float32
        if cv_embeddings.dtype != vaga_embedding.dtype:
            cv_embeddings_float = cv_embeddings.to(torch.float32)
            vaga_embedding_float = vaga_embedding.to(torch.float32)
        else:
            cv_embeddings_float = cv_embeddings
            vaga_embedding_float = vaga_embedding

        # calcular similaridade com embeddings do mesmo tipo
        scores = util.cos_sim(vaga_embedding_float, cv_embeddings_float).squeeze(0).cpu().numpy()
        top_idx = np.argsort(-scores)[:10]  # top 10

        candidatos = df_cvs.iloc[top_idx].copy()
        candidatos["score"] = scores[top_idx]

        # retornar candidatos como JSON
        data = [
            {
                "cv_id": int(row["cv_id"]), 
                "texto": str(row["cv_sugerido"]), 
                "score": float(row["score"])
            }
            for _, row in candidatos.iterrows()
        ]
        return Response(data)