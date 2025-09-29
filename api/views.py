from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework import viewsets, status
from rest_framework.pagination import PageNumberPagination
from ML.ML_bert import JobMatchingPipeline
import torch
from .models import Curriculo, Vaga
from .serializers import CurriculoSerializer, VagaSerializer
from sentence_transformers import util
import numpy as np

# Pipeline global
_pipeline = JobMatchingPipeline("bucket-tc5")

# Cache global
_cv_embeddings = None
_df_cvs = None
_df_vagas = None
_model = None
_metrics = None


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 12
    page_size_query_param = 'page_size'
    max_page_size = 100


class CurriculoViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Curriculo.objects.all()
    serializer_class = CurriculoSerializer
    pagination_class = StandardResultsSetPagination


class VagaViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Vaga.objects.all()
    serializer_class = VagaSerializer
    pagination_class = StandardResultsSetPagination

    @action(detail=True, methods=["get"])
    def predict(self, request, pk=None):
        global _cv_embeddings, _df_cvs, _df_vagas
        
        # Verificar se dados estão carregados
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
        
        # Verificar consistência
        if _df_cvs.empty or _df_vagas.empty:
            return Response(
                {"error": "DataFrames estão vazios. Recarregue os dados."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
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
            
            # Garantir modelo carregado
            _pipeline.embedding_processor._ensure_model_loaded()
            vaga_embedding = _pipeline.embedding_processor.encode_text(vaga_text)

            # Compatibilidade de dtype
            if _cv_embeddings.dtype != vaga_embedding.dtype:
                vaga_embedding = vaga_embedding.to(torch.float32)
                cv_embeddings = _cv_embeddings.to(torch.float32)
            else:
                cv_embeddings = _cv_embeddings

            # Calcular similaridade
            scores = util.cos_sim(vaga_embedding, cv_embeddings).squeeze(0).cpu().numpy()
            
            # Pegar parâmetro top_k da query (padrão 5)
            top_k = int(request.query_params.get('top_k', 5))
            top_idx = np.argsort(-scores)[:top_k]

            candidatos = _df_cvs.iloc[top_idx].copy()
            candidatos["score"] = scores[top_idx]

            data = [
                {
                    "cv_id": int(row["cv_id"]), 
                    "texto": str(row["cv_sugerido"])[:500] + "..." if len(str(row["cv_sugerido"])) > 500 else str(row["cv_sugerido"]), 
                    "score": float(row["score"]),
                    "match_percentage": round(float(row["score"]) * 100, 2)
                }
                for _, row in candidatos.iterrows()
            ]
            
            return Response({
                "vaga_id": pk,
                "vaga_titulo": vaga.titulo_vaga,
                "candidatos": data,
                "total_candidatos": len(data)
            })
            
        except Exception as e:
            return Response(
                {"error": f"Erro durante predição: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class LoadArtifactsView(APIView):
    def post(self, request):
        global _cv_embeddings, _df_cvs, _df_vagas, _pipeline
        try:
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
            
            _pipeline.embedding_processor._ensure_model_loaded()
            
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
            _cv_embeddings = None
            _df_cvs = None
            _df_vagas = None
            
            return Response({
                "error": str(e),
                "suggestion": "Verifique as credenciais AWS, conectividade de rede e se os arquivos existem no S3"
            }, status=500)


class TrainModelView(APIView):
    def post(self, request):
        global _model, _metrics, _cv_embeddings, _df_cvs, _df_vagas, _pipeline
        try:
            if _cv_embeddings is None or _df_cvs is None or _df_vagas is None:
                return Response({
                    "error": "Dados não carregados. Execute /api/ml/load/ primeiro."
                }, status=status.HTTP_400_BAD_REQUEST)
                
            params = request.data
            epochs = int(params.get("epochs", 10))
            batch_size = int(params.get("batch_size", 64))
            lr = float(params.get("learning_rate", 0.001))
            test_size = float(params.get("test_size", 0.2))
            top_k = int(params.get("top_k", 50))

            print(f"Iniciando treinamento com parâmetros:")
            print(f"- epochs: {epochs}")
            print(f"- batch_size: {batch_size}")
            print(f"- learning_rate: {lr}")
            print(f"- test_size: {test_size}")
            print(f"- top_k: {top_k}")

            _pipeline.embedding_processor._ensure_model_loaded()
            print("Modelo de embeddings carregado com sucesso")

            print("Executando pipeline de similaridade...")
            pairs_df = _pipeline.run_pipeline(top_k=top_k)
            
            print(f"Pipeline executado: {len(pairs_df)} pares gerados")
            
            print("Iniciando treinamento do modelo...")
            _model, trainer, _metrics = _pipeline.train_model(
                pairs_df, 
                epochs=epochs, 
                batch_size=batch_size, 
                lr=lr, 
                test_size=test_size
            )

            print("Treinamento concluído com sucesso!")

            return Response({
                "message": "Modelo treinado com sucesso",
                "parameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "test_size": test_size,
                    "top_k": top_k
                },
                "data_info": {
                    "total_pairs": len(pairs_df),
                    "positive_pairs": int(pairs_df["label"].sum()),
                    "negative_pairs": int(len(pairs_df) - pairs_df["label"].sum())
                },
                "metrics": {
                    "accuracy": _metrics.get("accuracy", 0),
                    "avg_precision_score": _metrics.get("avg_precision_score", 0),
                    "macro_avg_f1_score": _metrics.get("macro_avg_f1_score", 0),
                    "mlflow_run_id": _metrics.get("mlflow_run_id", "N/A")
                }
            })
            
        except Exception as e:
            print(f"Erro durante treinamento: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return Response({
                "error": f"Erro durante treinamento: {str(e)}",
                "suggestion": "Verifique os logs para mais detalhes"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MetricsView(APIView):
    def get(self, request):
        global _metrics
        if _metrics is None:
            return Response({"error": "Modelo não treinado. Rode /api/ml/train/ primeiro."}, status=400)
        return Response({"metrics": _metrics})


class HealthView(APIView):
    def get(self, request):
        global _cv_embeddings, _df_cvs, _df_vagas, _model, _metrics, _pipeline
        
        status_ok = True
        errors = []
        details = {}

        if _pipeline is None:
            status_ok = False
            errors.append("Pipeline não inicializada")
        else:
            details["pipeline"] = "OK"

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
            
        if _df_cvs is not None and _df_vagas is not None:
            details["data_counts"] = {
                "curriculos": len(_df_cvs),
                "vagas": len(_df_vagas)
            }
            
        if _cv_embeddings is not None:
            details["embeddings_shape"] = list(_cv_embeddings.shape)

        embedding_model_loaded = (_pipeline is not None and 
                                _pipeline.embedding_processor is not None and 
                                _pipeline.embedding_processor.model is not None)
        
        details["embedding_model_loaded"] = embedding_model_loaded
        
        if not embedding_model_loaded:
            errors.append("Modelo de embeddings não carregado")

        return Response({
            "status": "ok" if status_ok else "error",
            "errors": errors,
            "details": details,
            "ready_for_prediction": (_cv_embeddings is not None and 
                                   _df_cvs is not None and 
                                   _df_vagas is not None),
            "ready_for_training": (_df_cvs is not None and 
                                 _df_vagas is not None and
                                 embedding_model_loaded)
        })
    
from django.shortcuts import render

def portal_vagas(request):
    return render(request, 'vagas.html')