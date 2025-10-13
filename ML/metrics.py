# ML/metrics.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# Métricas básicas
predictions_total = Counter('ml_predictions_total', 'Total de predições', ['model_version'])
prediction_score = Histogram('ml_prediction_score', 'Score das predições', ['model_version'])
training_accuracy = Gauge('ml_training_accuracy', 'Acurácia do modelo', ['model_version'])
training_loss = Gauge('ml_training_loss', 'Loss do treinamento', ['model_version'])

_prometheus_started = False

def start_prometheus_server(port=8001):
    """Inicia servidor Prometheus (apenas uma vez)"""
    global _prometheus_started
    if _prometheus_started:
        print(f"⚠ Prometheus já está rodando na porta {port}")
        return
    
    try:
        start_http_server(port)
        _prometheus_started = True
        print(f"✓ Prometheus rodando na porta {port}")
        print(f"✓ Acesse: http://localhost:{port}/metrics")
    except OSError as e:
        print(f"⚠ Erro ao iniciar Prometheus: {e}")
        print(f"   Porta {port} pode estar em uso")

def log_prediction(score, model_version="v1"):
    """Registra uma predição"""
    predictions_total.labels(model_version=model_version).inc()
    prediction_score.labels(model_version=model_version).observe(float(score))

def log_training_metrics(accuracy, loss, model_version="v1"):
    """Registra métricas de treinamento"""
    training_accuracy.labels(model_version=model_version).set(float(accuracy))
    training_loss.labels(model_version=model_version).set(float(loss))