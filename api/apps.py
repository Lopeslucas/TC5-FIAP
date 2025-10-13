# api/apps.py

from django.apps import AppConfig

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    
    def ready(self):
        """Executado quando Django inicializa"""
        # Inicia Prometheus apenas uma vez
        from ML.metrics import start_prometheus_server
        import os
        
        # Evita iniciar duas vezes (runserver inicia processo duplicado)
        if os.environ.get('RUN_MAIN') == 'true':
            try:
                start_prometheus_server(8001)
                print("✓ Prometheus metrics disponível em http://localhost:8001/metrics")
            except OSError:
                print("⚠ Prometheus já está rodando")