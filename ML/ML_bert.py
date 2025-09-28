#ML/ML_bert.py

import os
import io
import boto3
import torch
import numpy as np
import pandas as pd
import argparse
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
        self.model = SentenceTransformer(model_name, device=self.device)
        self.s3_loader = s3_loader

    def get_cv_embeddings(self, cv_texts, s3_path="gold/cv_embeddings.npy", dtype=np.float32):
        try:
            obj = self.s3_loader.s3.get_object(Bucket=self.s3_loader.bucket_name, Key=s3_path)
            array = np.load(io.BytesIO(obj["Body"].read()))
            print(f"Embeddings de CV carregados de s3://{self.s3_loader.bucket_name}/{s3_path}")
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        except self.s3_loader.s3.exceptions.NoSuchKey:
            print("Embeddings de CV não encontrados. Gerando e salvando...")
            embeddings = self.model.encode(cv_texts, convert_to_tensor=True).to(torch.float32)
            buffer = io.BytesIO()
            np.save(buffer, embeddings.cpu().numpy().astype(dtype))
            buffer.seek(0)
            self.s3_loader.upload_bytes(buffer.read(), s3_path)
            return embeddings

    def encode_text(self, text):
        return self.model.encode(text, convert_to_tensor=True)

# Similaridade
class SimilarityCalculator:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.scaler = MinMaxScaler()

    def calculate_top_k_pairs(self, df_vagas, df_cvs, cv_embeddings, embedding_processor, top_k=50):
        vaga_texts = (df_vagas["titulo_vaga"].astype(str) + " " + df_vagas["principais_atividades"].astype(str)).tolist()
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

# Treinador
class ModelTrainer:
    def __init__(self, model, pos_weight=None, lr=0.001):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def prepare_data(self, pairs_df, features=["semantic_norm", "tfidf_norm"], test_size=0.2, random_state=50):
        X = pairs_df[features].values.astype("float32")
        y = pairs_df["label"].values.astype("float32")
        groups = pairs_df["job_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=64):
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        return train_loader, test_loader

    def train(self, train_loader, epochs=10):
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
            
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            # Log da loss de treino para o MLflow
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_test)).squeeze(1)
            y_probs = torch.sigmoid(logits).numpy()
            y_pred = (y_probs >= 0.5).astype(int)

        # Retorna métricas como dicionário para facilitar o log
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        avg_prec = average_precision_score(y_test, y_probs)
        
        # Cria e salva a figura da curva PR para loggar como artefato
        fig = plt.figure(figsize=(6, 4))
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        plt.plot(recall, precision, label=f"AP={avg_prec:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        
        # Não exibe a figura, apenas a retorna para ser salva pelo MLflow
        # plt.show() -> substituído
        
        metrics = {
            "accuracy": report_dict["accuracy"],
            "avg_precision_score": avg_prec,
            "macro_avg_precision": report_dict["macro avg"]["precision"],
            "macro_avg_recall": report_dict["macro avg"]["recall"],
            "macro_avg_f1_score": report_dict["macro avg"]["f1-score"],
            "weighted_avg_precision": report_dict["weighted avg"]["precision"],
            "weighted_avg_recall": report_dict["weighted avg"]["recall"],
            "weighted_avg_f1_score": report_dict["weighted avg"]["f1-score"],
        }
        
        return metrics, fig

# Pipeline
class JobMatchingPipeline:
    def __init__(self, bucket_name, prefix="silver/", model_name="all-MiniLM-L6-v2"):
        self.s3_loader = S3DataLoader(bucket_name)
        self.embedding_processor = EmbeddingProcessor(self.s3_loader, model_name)
        self.similarity_calc = SimilarityCalculator()
        self.label_generator = LabelGenerator()
        self.prefix = prefix

    def load_data(self, cv_key="cv_filtred.csv", vagas_key="vagas_filtred.csv", prefix=None):
        if prefix is None:
            prefix = self.prefix
        
        df_cvs = self.s3_loader.load_csv(prefix + cv_key)
        df_vagas = self.s3_loader.load_csv(prefix + vagas_key)
        df_cvs = df_cvs[["id", "cv_sugerido"]].rename(columns={"id": "cv_id"})
        df_vagas = df_vagas[["id", "titulo_vaga", "principais_atividades"]].rename(columns={"id": "job_id"})
        return df_cvs, df_vagas

    def run_pipeline(self, args):
        df_cvs, df_vagas = self.load_data(args.cv_input_key, args.vagas_input_key, args.s3_prefix)
        cv_embeddings = self.embedding_processor.get_cv_embeddings(
            df_cvs["cv_sugerido"].astype(str).tolist(), 
            s3_path=args.cv_embeddings_path
        )
        pairs_df = self.similarity_calc.calculate_top_k_pairs(df_vagas, df_cvs, cv_embeddings, self.embedding_processor, args.top_k)
        pairs_df["tfidf_score"] = self.similarity_calc.calculate_tfidf_scores(pairs_df, df_vagas, df_cvs)
        pairs_df = self.similarity_calc.normalize_and_combine_scores(pairs_df, args.semantic_weight, args.tfidf_weight)
        pairs_df["label"] = self.label_generator.create_top_percentile_labels(pairs_df, args.label_percentile)

        buffer_csv = io.BytesIO()
        pairs_df.to_csv(buffer_csv, index=False)
        self.s3_loader.upload_bytes(buffer_csv.getvalue(), self.prefix + args.output_key_csv)

        buffer_parquet = io.BytesIO()
        pairs_df.to_parquet(buffer_parquet, index=False)
        self.s3_loader.upload_bytes(buffer_parquet.getvalue(), self.prefix + args.output_key_parquet)

        print(f"Pipeline concluída com {len(pairs_df)} pares")
        return pairs_df
    
    def train_model(self, pairs_df, args):
        model = MatchNet(input_dim=2, hidden_dims=args.hidden_dims)
        n_pos = pairs_df["label"].sum()
        n_neg = len(pairs_df) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32) if n_pos > 0 else None

        trainer = ModelTrainer(model, pos_weight=pos_weight, lr=args.learning_rate)
        X_train, X_test, y_train, y_test = trainer.prepare_data(pairs_df, test_size=args.test_size, random_state=args.random_state)
        train_loader, _ = trainer.create_data_loaders(X_train, X_test, y_train, y_test, batch_size=args.batch_size)
        
        print("Iniciando treinamento do modelo...")
        trainer.train(train_loader, args.epochs)
        
        print("Avaliando o modelo...")
        metrics, pr_curve_fig = trainer.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")

        fig_path = "precision_recall_curve.png"
        pr_curve_fig.savefig(fig_path)
        
        # Log de tudo para o MLflow
        print("Registrando resultados no MLflow...")
        mlflow.log_params(vars(args))
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(fig_path)
        plt.close(pr_curve_fig) # Fecha a figura para liberar memória
        
        print("Registro no MLflow concluído.")
        return model, trainer

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Job Matching com BERT e PyTorch.")

    # Argumentos do MLflow
    parser.add_argument("--mlflow-experiment-name", type=str, default="Job Matching Experiment", help="Nome do experimento no MLflow.")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Nome da execução (run) no MLflow.")

    # Argumentos S3 e Dados
    parser.add_argument(
        "--bucket_name",
        type=str,
        default="bucket-tc5",
        help="Nome do bucket S3."
    )

    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="silver/",
        help="Prefixo (pasta) no S3 para os dados."
    )

    parser.add_argument(
        "--cv_input_key",
        type=str,
        default="cv_filtred.csv",
        help="Nome do arquivo de CVs."
    )

    parser.add_argument(
        "--vagas_input_key",
        type=str,
        default="vagas_filtred.csv",
        help="Nome do arquivo de vagas."
    )

    parser.add_argument(
        "--cv_embeddings_path",
        type=str,
        default="gold/cv_embeddings.npy",
        help="Caminho para salvar/carregar os embeddings dos CVs."
    )

    parser.add_argument(
        "--output_key_csv",
        type=str,
        default="dataset_topk.csv",
        help="Nome do arquivo de saída CSV."
    )

    parser.add_argument(
        "--output_key_parquet",
        type=str,
        default="dataset_topk.parquet",
        help="Nome do arquivo de saída Parquet."
    )
    
    # Argumentos de Processamento e Geração de Features
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Nome do modelo SentenceTransformer."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Número de CVs a serem considerados para cada vaga (top-k)."
    )

    parser.add_argument(
        "--semantic_weight",
        type=float,
        default=0.7,
        help="Peso para o score semântico."
    )

    parser.add_argument(
        "--tfidf_weight",
        type=float,
        default=0.3,
        help="Peso para o score TF-IDF."
    )

    parser.add_argument(
        "--label_percentile",
        type=float,
        default=0.9,
        help="Percentil para definir o label positivo (match)."
    )
    
    # Argumentos de Treinamento do Modelo
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs='+',
        default=[16, 8],
        help="Dimensões das camadas ocultas da rede neural."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Taxa de aprendizado (learning rate) do otimizador."
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporção do dataset para o conjunto de teste."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Tamanho do lote (batch size) para o treinamento."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas para o treinamento."
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=50,
        help="Semente aleatória para reprodutibilidade."
    )

    args = parser.parse_args()

    load_dotenv()

    # Configuração do experimento no MLflow
    mlflow.set_experiment(args.mlflow_experiment_name)

    # Inicia uma execução (run) do MLflow
    with mlflow.start_run(run_name=args.mlflow_run_name):
        pipeline = JobMatchingPipeline(
            bucket_name=args.bucket_name,
            prefix=args.s3_prefix,
            model_name=args.model_name
        )
        pairs_df = pipeline.run_pipeline(args)
        pipeline.train_model(pairs_df, args)

if __name__ == "__main__":
    main()