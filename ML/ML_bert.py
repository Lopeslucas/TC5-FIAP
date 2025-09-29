#ML/ML_bert.py

import os
import io
import boto3
import torch
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Importações MLflow
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import tempfile
import joblib


# S3 Loader
class S3DataLoader:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )

    def load_csv(self, key):
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        return pd.read_csv(obj["Body"], sep=";")

    def upload_bytes(self, data: bytes, s3_path: str):
        self.s3.upload_fileobj(io.BytesIO(data), self.bucket_name, s3_path)
        print(f"Upload concluído: s3://{self.bucket_name}/{s3_path}")


# Embeddings
class EmbeddingProcessor:
    def __init__(self, s3_loader: S3DataLoader, model_name="all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.s3_loader = s3_loader

    def _ensure_model_loaded(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def get_cv_embeddings(self, cv_texts, s3_path="gold/cv_embeddings.npy", dtype=np.float32):
        try:
            obj = self.s3_loader.s3.get_object(Bucket=self.s3_loader.bucket_name, Key=s3_path)
            array = np.load(io.BytesIO(obj["Body"].read()))
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        except self.s3_loader.s3.exceptions.NoSuchKey:

            self._ensure_model_loaded()
            embeddings = self.model.encode(cv_texts, convert_to_tensor=True).to(torch.float32)
            buffer = io.BytesIO()
            np.save(buffer, embeddings.cpu().numpy().astype(dtype))
            buffer.seek(0)
            self.s3_loader.upload_bytes(buffer.read(), s3_path)
            return embeddings

    def encode_text(self, text):
        self._ensure_model_loaded()
        return self.model.encode(text, convert_to_tensor=True)


# Similaridade
class SimilarityCalculator:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.scaler = MinMaxScaler()

    def calculate_top_k_pairs(self, df_vagas, df_cvs, cv_embeddings, embedding_processor, top_k=50):
        vaga_texts = (df_vagas["titulo_vaga"].astype(str) + " " + df_vagas["principais_atividades"].astype(str)).tolist()
        
        # Garantir que o modelo está carregado antes de usar
        embedding_processor._ensure_model_loaded()
        vaga_embeddings = embedding_processor.model.encode(vaga_texts, convert_to_tensor=True)
        vaga_embeddings = vaga_embeddings.to(cv_embeddings.dtype)

        hits = util.semantic_search(vaga_embeddings, cv_embeddings, top_k=top_k)
        all_pairs = []

        for i, job_id in enumerate(df_vagas["job_id"]):
            for h in hits[i]:
                all_pairs.append({
                    "job_id": job_id,
                    "cv_id": df_cvs.iloc[h["corpus_id"]]["cv_id"],
                    "semantic_score": float(h["score"])
                })

        return pd.DataFrame(all_pairs)

    def calculate_tfidf_scores(self, pairs_df, df_vagas, df_cvs):
        vaga_texts = (df_vagas["titulo_vaga"].astype(str) + " " + df_vagas["principais_atividades"].astype(str)).tolist()
        cv_texts = df_cvs["cv_sugerido"].astype(str).tolist()
        all_texts = vaga_texts + cv_texts

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        vaga_tfidf = tfidf_matrix[:len(vaga_texts)]
        cv_tfidf = tfidf_matrix[len(vaga_texts):]

        scores_matrix = cosine_similarity(vaga_tfidf, cv_tfidf)
        tfidf_scores = [scores_matrix[df_vagas.index[df_vagas["job_id"] == row.job_id][0],
                                        df_cvs.index[df_cvs["cv_id"] == row.cv_id][0]]
                        for _, row in pairs_df.iterrows()]
        return tfidf_scores

    def normalize_and_combine_scores(self, pairs_df, semantic_weight=0.7, tfidf_weight=0.3):
        pairs_df[["semantic_norm", "tfidf_norm"]] = self.scaler.fit_transform(
            pairs_df[["semantic_score", "tfidf_score"]]
        )
        pairs_df["final_score"] = semantic_weight * pairs_df["semantic_norm"] + tfidf_weight * pairs_df["tfidf_norm"]
        return pairs_df


# Label Generator
class LabelGenerator:
    @staticmethod
    def create_top_percentile_labels(pairs_df, percentile=0.9):
        labels = []
        for _, group in pairs_df.groupby("job_id"):
            threshold = group["final_score"].quantile(percentile)
            labels.extend((group["final_score"] >= threshold).astype(int).tolist())
        return labels


# Modelo
class MatchNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Treinador com MLflow
class ModelTrainer:
    def __init__(self, model, pos_weight=None, lr=0.001):
        if pos_weight is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.lr = lr
        self.pos_weight = pos_weight

    def prepare_data(self, pairs_df, features=["semantic_norm", "tfidf_norm"], test_size=0.2):
        X = pairs_df[features].values.astype("float32")
        y = pairs_df["label"].values.astype("float32")
        groups = pairs_df["job_id"].values

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=64):
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        return train_loader, test_loader

    def train(self, train_loader, epochs=10):
        train_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(xb).squeeze(1)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            
            epoch_loss = total_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            # Log da loss por época no MLflow
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return train_losses

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_test)).squeeze(1)
            y_probs = torch.sigmoid(logits).numpy()
            y_pred = (y_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # curva precision-recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
        avg_prec = average_precision_score(y_test, y_probs)

        # Salvar gráfico PR curve no MLflow
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"AP={avg_prec:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        
        # Salvar figura temporariamente e logar no MLflow
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(tmp_file.name, "plots")
        plt.close()

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Adicionar valores na matriz
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center", 
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(tmp_file.name, "plots")
        plt.close()

        return acc, report, y_probs, y_pred


# Pipeline com MLflow
class JobMatchingPipeline:
    def __init__(self, bucket_name, prefix="silver/", mlflow_experiment_name="job-matching-v2"):
        self.s3_loader = S3DataLoader(bucket_name)
        self.embedding_processor = EmbeddingProcessor(self.s3_loader)
        self.similarity_calc = SimilarityCalculator()
        self.label_generator = LabelGenerator()
        self.bucket_name = bucket_name
        self.prefix = prefix
        
        # Configurar MLflow
        mlflow.set_experiment(mlflow_experiment_name)
        
    def load_data(self):
        df_cvs = self.s3_loader.load_csv(self.prefix + "cv_filtred.csv")
        df_vagas = self.s3_loader.load_csv(self.prefix + "vagas_filtred.csv")
        df_cvs = df_cvs[["id", "cv_sugerido"]].rename(columns={"id": "cv_id"})
        df_vagas = df_vagas[["id", "titulo_vaga", "principais_atividades"]].rename(columns={"id": "job_id"})
        return df_cvs, df_vagas

    def run_pipeline(self, top_k=50):
        df_cvs, df_vagas = self.load_data()
        
        # Log parâmetros do pipeline
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("cv_count", len(df_cvs))
        mlflow.log_param("vaga_count", len(df_vagas))
        mlflow.log_param("embedding_model", self.embedding_processor.model_name)
        
        cv_embeddings = self.embedding_processor.get_cv_embeddings(df_cvs["cv_sugerido"].astype(str).tolist())

        pairs_df = self.similarity_calc.calculate_top_k_pairs(df_vagas, df_cvs, cv_embeddings, self.embedding_processor, top_k)
        pairs_df["tfidf_score"] = self.similarity_calc.calculate_tfidf_scores(pairs_df, df_vagas, df_cvs)
        pairs_df = self.similarity_calc.normalize_and_combine_scores(pairs_df)
        pairs_df["label"] = self.label_generator.create_top_percentile_labels(pairs_df)

        # Log estatísticas dos dados
        mlflow.log_metric("total_pairs", len(pairs_df))
        mlflow.log_metric("positive_pairs", pairs_df["label"].sum())
        mlflow.log_metric("negative_pairs", len(pairs_df) - pairs_df["label"].sum())
        mlflow.log_metric("positive_ratio", pairs_df["label"].mean())

        # Salvar datasets
        buffer = io.BytesIO()
        pairs_df.to_csv(buffer, index=False)
        self.s3_loader.upload_bytes(buffer.getvalue(), "silver/dataset_topk.csv")

        buffer = io.BytesIO()
        pairs_df.to_parquet(buffer, index=False)
        self.s3_loader.upload_bytes(buffer.getvalue(), "silver/dataset_topk.parquet")

        # Salvar dataset como artefato no MLflow
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            pairs_df.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, "datasets")

        print(f"Pipeline concluída com {len(pairs_df)} pares")
        return pairs_df

    def train_model(self, pairs_df, epochs=10, test_size=0.2, batch_size=64, lr=0.001, 
                   semantic_weight=0.7, tfidf_weight=0.3, hidden_dims=[16, 8], seed=42):
            
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Finalizar qualquer run ativa antes de começar uma nova
        try:
            mlflow.end_run()
        except:
            pass  # Ignorar se não há run ativa
        
        with mlflow.start_run() as run:
            
            # Log de todos os parâmetros
            mlflow.log_params({
                "epochs": epochs,
                "test_size": test_size,
                "batch_size": batch_size,
                "learning_rate": lr,
                "semantic_weight": semantic_weight,
                "tfidf_weight": tfidf_weight,
                "hidden_dims": str(hidden_dims),
                "model_architecture": "MatchNet",
                "optimizer": "Adam",
                "loss_function": "BCEWithLogitsLoss",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "random_seed": seed
            })
            
            model = MatchNet(input_dim=2, hidden_dims=hidden_dims)

            # Balanceamento
            n_pos = pairs_df["label"].sum()
            n_neg = len(pairs_df) - n_pos
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
            
            mlflow.log_param("pos_weight", pos_weight.item())

            trainer = ModelTrainer(model, pos_weight=pos_weight, lr=lr)
            X_train, X_test, y_train, y_test = trainer.prepare_data(pairs_df, test_size=test_size)
            train_loader, test_loader = trainer.create_data_loaders(X_train, X_test, y_train, y_test, batch_size=batch_size)
            
            # Log tamanhos dos conjuntos
            mlflow.log_metrics({
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum())
            })
            
            # Treinamento com logging
            print("Iniciando treinamento...")
            train_losses = trainer.train(train_loader, epochs)

            # Avaliação
            print("Avaliando modelo...")
            acc, report_dict, y_probs, y_pred = trainer.evaluate(X_test, y_test)
            avg_prec = average_precision_score(y_test, y_probs)
            cm = confusion_matrix(y_test, y_pred).tolist()

            # Log de todas as métricas
            # Tentar com string e inteiro para as classes
            class_0_key = "0" if "0" in report_dict else 0
            class_1_key = "1" if "1" in report_dict else 1
            
            metrics = {
                "accuracy": acc,
                "avg_precision_score": avg_prec,
                "macro_avg_precision": report_dict["macro avg"]["precision"],
                "macro_avg_recall": report_dict["macro avg"]["recall"],
                "macro_avg_f1_score": report_dict["macro avg"]["f1-score"],
                "weighted_avg_precision": report_dict["weighted avg"]["precision"],
                "weighted_avg_recall": report_dict["weighted avg"]["recall"],
                "weighted_avg_f1_score": report_dict["weighted avg"]["f1-score"],
                "final_train_loss": train_losses[-1] if train_losses else 0.0
            }
            
            # Adicionar métricas por classe se existirem
            if class_0_key in report_dict:
                metrics.update({
                    "class_0_precision": report_dict[class_0_key]["precision"],
                    "class_0_recall": report_dict[class_0_key]["recall"], 
                    "class_0_f1_score": report_dict[class_0_key]["f1-score"]
                })
            
            if class_1_key in report_dict:
                metrics.update({
                    "class_1_precision": report_dict[class_1_key]["precision"],
                    "class_1_recall": report_dict[class_1_key]["recall"],
                    "class_1_f1_score": report_dict[class_1_key]["f1-score"]
                })
            
            mlflow.log_metrics(metrics)

            # Salvar modelo no MLflow
            # Criar signature para o modelo
            signature = infer_signature(X_test, y_probs)
            mlflow.pytorch.log_model(
                model, 
                "model",
                signature=signature,
                input_example=X_test[:5] if len(X_test) > 5 else X_test
            )
            
            # Salvar scaler como artefato
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                joblib.dump(self.similarity_calc.scaler, tmp_file.name)
                mlflow.log_artifact(tmp_file.name, "preprocessors/scaler.pkl")
                
            # Salvar vectorizer como artefato
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                joblib.dump(self.similarity_calc.tfidf_vectorizer, tmp_file.name)
                mlflow.log_artifact(tmp_file.name, "preprocessors/tfidf_vectorizer.pkl")

            # Métricas completas para S3
            full_metrics = {
                "n_samples": len(pairs_df),
                "n_positive": int(n_pos),
                "n_negative": int(n_neg),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "confusion_matrix": cm,
                "classification_report": report_dict,
                "mlflow_run_id": run.info.run_id,
                "mlflow_experiment_id": run.info.experiment_id,
                **metrics
            }

            # Salvar métricas no S3
            buffer = io.BytesIO()
            buffer.write(json.dumps(full_metrics, indent=2).encode("utf-8"))
            buffer.seek(0)
            self.s3_loader.upload_bytes(buffer.read(), "gold/metrics.json")
            
            # Log das métricas como artefato também
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                with open(tmp_file.name, 'w') as f:
                    json.dump(full_metrics, f, indent=2)
                mlflow.log_artifact(tmp_file.name, "metrics/full_metrics.json")

            # Log informações adicionais
            mlflow.set_tags({
                "model_type": "job_matching",
                "framework": "pytorch",
                "data_source": "s3",
                "bucket": self.bucket_name
            })

            print(f"Treinamento concluído!")
            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Average Precision: {avg_prec:.4f}")

            return model, trainer, full_metrics


if __name__ == "__main__":
    load_dotenv()
    
    pipeline = JobMatchingPipeline("bucket-tc5", mlflow_experiment_name="job-matching-experiment")
    pairs_df = pipeline.run_pipeline(top_k=50)
    model, trainer, metrics = pipeline.train_model(
        pairs_df, 
        epochs=10,
        lr=0.001,
        batch_size=64,
        hidden_dims=[16, 8]
    )