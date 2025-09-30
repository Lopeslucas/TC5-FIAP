# ML/monitoring.py
import json
import io
from datetime import datetime

def log_predictions_to_s3(pairs_df, y_probs, model_version, s3_loader):
    """Salva predições no S3 (simples)"""
    try:
        predictions = []
        for i in range(len(y_probs)):
            predictions.append({
                "job_id": int(pairs_df.iloc[i]["job_id"]),
                "cv_id": int(pairs_df.iloc[i]["cv_id"]),
                "score": float(y_probs[i]),
                "timestamp": datetime.now().isoformat()
            })
        
        # Salva no S3
        buffer = io.BytesIO()
        buffer.write(json.dumps(predictions, indent=2).encode("utf-8"))
        buffer.seek(0)
        s3_loader.upload_bytes(buffer.read(), f"gold/predictions_{model_version}.json")
        
        print(f"✓ {len(predictions)} predições salvas no S3")
        return True
    except Exception as e:
        print(f"⚠ Erro ao salvar predições: {e}")
        return False

def log_training_summary(metrics, model_version, s3_loader):
    """Salva resumo do treinamento no S3"""
    try:
        summary = {
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        buffer = io.BytesIO()
        buffer.write(json.dumps(summary, indent=2).encode("utf-8"))
        buffer.seek(0)
        s3_loader.upload_bytes(buffer.read(), f"gold/training_summary_{model_version}.json")
        
        print(f"✓ Resumo do treinamento salvo no S3")
        return True
    except Exception as e:
        print(f"⚠ Erro ao salvar resumo: {e}")
        return False