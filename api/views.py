from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework import viewsets, status
from ML.ML_bert import JobMatchingPipeline
import torch
from .models import Curriculo, Vaga
from .serializers import CurriculoSerializer, VagaSerializer
from sentence_transformers import util
import numpy as np

# Pipeline global
_pipeline = JobMatchingPipeline("bucket-tc5")

# Cache global - inicializar com None explicitamente
_cv_embeddings = None
_df_cvs = None
_df_vagas = None
_model = None
_metrics = None

class CurriculoViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Curriculo.objects.all()
    serializer_class = CurriculoSerializer


class VagaViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Vaga.objects.all()
    serializer_class = VagaSerializer

    @action(detail=True, methods=["get"])
    def predict(self, request, pk=None):
        global _cv_embeddings, _df_cvs, _df_vagas
        
        # Verificar se TODOS os dados necessários estão carregados
        if _cv_embeddings is None or _df_cvs is None or _df_vagas is None:
            missing = []
            if _cv_embeddings is None:
                missing.append("embeddings")
            if _df_cvs is None:
                missing.append("curriculos")
            if _df_vagas is None:
                missing.append("vagas")
            
            return Response(
                {
                    "error": f"Dados não carregados: {', '.join(missing)}. Rode /api/ml/load/ primeiro.",
                    "loaded_data": {
                        "embeddings": _cv_embeddings is not None,
                        "curriculos": _df_cvs is not None,
                        "vagas": _df_vagas is not None
                    }
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Verificar se os DataFrames não estão vazios
        if _df_cvs.empty or _df_vagas.empty:
            return Response(
                {"error": "DataFrames estão vazios. Recarregue os dados."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Verificar se os embeddings têm o tamanho correto
        if len(_cv_embeddings) != len(_df_cvs):
            return Response(
                {
                    "error": f"Inconsistência nos dados: {len(_cv_embeddings)} embeddings vs {len(_df_cvs)} currículos",
                    "suggestion": "Recarregue os dados com /api/ml/load/"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            vaga = self.get_object()
            vaga_text = f"{vaga.titulo_vaga} {vaga.principais_atividades}"
            
            # Garantir que o modelo de embeddings está carregado
            vaga_embedding = _pipeline.embedding_processor.encode_text(vaga_text)

            # Compatibilidade de dtype
            if _cv_embeddings.dtype != vaga_embedding.dtype:
                vaga_embedding = vaga_embedding.to(torch.float32)
                cv_embeddings = _cv_embeddings.to(torch.float32)
            else:
                cv_embeddings = _cv_embeddings

            scores = util.cos_sim(vaga_embedding, cv_embeddings).squeeze(0).cpu().numpy()
            top_idx = np.argsort(-scores)[:10]

            candidatos = _df_cvs.iloc[top_idx].copy()
            candidatos["score"] = scores[top_idx]

            data = [
                {
                    "cv_id": int(row["cv_id"]), 
                    "texto": str(row["cv_sugerido"]), 
                    "score": float(row["score"])
                }
                for _, row in candidatos.iterrows()
            ]
            
            return Response({
                "vaga_id": pk,
                "candidatos": data,
                "total_candidatos": len(data)
            })
            
        except Exception as e:
            return Response(
                {"error": f"Erro durante predição: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Load artefatos (embeddings, datasets)
class LoadArtifactsView(APIView):
    def post(self, request):
        global _cv_embeddings, _df_cvs, _df_vagas
        try:
            # Resetar variáveis globais no início
            _cv_embeddings = None
            _df_cvs = None
            _df_vagas = None
            
            print("Carregando dados do S3...")
            _df_cvs, _df_vagas = _pipeline.load_data()
            
            if _df_cvs.empty or _df_vagas.empty:
                return Response({
                    "error": "Arquivos S3 estão vazios ou não encontrados.",
                    "details": {
                        "curriculos_carregados": len(_df_cvs),
                        "vagas_carregadas": len(_df_vagas)
                    }
                }, status=400)

            print(f"Dados carregados: {len(_df_cvs)} CVs, {len(_df_vagas)} vagas")
            print("Gerando embeddings...")
            
            _cv_embeddings = _pipeline.embedding_processor.get_cv_embeddings(
                _df_cvs["cv_sugerido"].astype(str).tolist()
            )

            if _cv_embeddings is None or len(_cv_embeddings) == 0:
                return Response({
                    "error": "Não foi possível gerar embeddings.",
                    "details": "Verifique os dados de entrada e a conexão com o S3"
                }, status=500)

            print(f"Embeddings gerados: {_cv_embeddings.shape}")

            return Response({
                "message": "Artefatos carregados com sucesso",
                "curriculos": len(_df_cvs),
                "vagas": len(_df_vagas),
                "embeddings_shape": list(_cv_embeddings.shape),
                "device": str(_cv_embeddings.device) if hasattr(_cv_embeddings, 'device') else "unknown"
            })
            
        except Exception as e:
            # Resetar em caso de erro
            _cv_embeddings = None
            _df_cvs = None
            _df_vagas = None
            
            return Response({
                "error": str(e),
                "suggestion": "Verifique as credenciais AWS, conectividade de rede e se os arquivos existem no S3"
            }, status=500)


# Train model (parametrizado)
class TrainModelView(APIView):
    def post(self, request):
        global _model, _metrics
        try:
            # Verificar se os dados estão carregados antes do treinamento
            if _cv_embeddings is None or _df_cvs is None or _df_vagas is None:
                return Response({
                    "error": "Dados não carregados. Execute /api/ml/load/ primeiro."
                }, status=status.HTTP_400_BAD_REQUEST)
                
            params = request.data
            epochs = int(params.get("epochs", 10))
            batch_size = int(params.get("batch_size", 64))
            lr = float(params.get("learning_rate", 0.001))
            test_size = float(params.get("test_size", 0.2))

            # treinar pipeline completo
            pairs_df = _pipeline.run_pipeline(top_k=int(params.get("top_k", 50)))
            _model, trainer, _metrics = _pipeline.train_model(
                pairs_df, epochs=epochs, batch_size=batch_size, lr=lr, test_size=test_size
            )

            return Response({
                "message": "Modelo treinado com sucesso",
                "epochs": epochs,
                "batch_size": batch_size,
                "metrics": _metrics
            })
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Metrics
class MetricsView(APIView):
    def get(self, request):
        global _metrics
        if _metrics is None:
            return Response({"error": "Modelo não treinado. Rode /api/ml/train/ primeiro."}, status=400)
        return Response({"metrics": _metrics})


# Health (readiness/liveness) - melhorado
class HealthView(APIView):
    def get(self, request):
        global _cv_embeddings, _df_cvs, _df_vagas, _model, _metrics
        
        status_ok = True
        errors = []
        details = {}

        # Verificar pipeline carregada
        if _pipeline is None:
            status_ok = False
            errors.append("Pipeline não inicializada")
        else:
            details["pipeline"] = "OK"

        # Verificar dados carregados
        details["data_status"] = {
            "embeddings": _cv_embeddings is not None,
            "curriculos": _df_cvs is not None,
            "vagas": _df_vagas is not None,
            "model": _model is not None,
            "metrics": _metrics is not None
        }
        
        if _cv_embeddings is None:
            errors.append("Embeddings não carregados")
            
        if _df_cvs is None:
            errors.append("Dados de currículos não carregados")
            
        if _df_vagas is None:
            errors.append("Dados de vagas não carregados")
            
        # Informações adicionais se os dados estão carregados
        if _df_cvs is not None and _df_vagas is not None:
            details["data_counts"] = {
                "curriculos": len(_df_cvs),
                "vagas": len(_df_vagas)
            }
            
        if _cv_embeddings is not None:
            details["embeddings_shape"] = list(_cv_embeddings.shape)

        return Response({
            "status": "ok" if status_ok else "error",
            "errors": errors,
            "details": details,
            "ready_for_prediction": (_cv_embeddings is not None and 
                                   _df_cvs is not None and 
                                   _df_vagas is not None)
        })