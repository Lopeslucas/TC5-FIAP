#ML/ML_bert.py

import os
import io
import boto3
import torch
import numpy as np
import pandas as pd
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
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        except self.s3_loader.s3.exceptions.NoSuchKey:
            embeddings = self.model.encode(cv_texts, convert_to_tensor=True).to(torch.float32 if dtype==np.float32 else torch.float32)
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
        if pos_weight is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

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
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_test)).squeeze(1)
            y_probs = torch.sigmoid(logits).numpy()
            y_pred = (y_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # curva precision-recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
        avg_prec = average_precision_score(y_test, y_probs)

        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f"AP={avg_prec:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()

        return acc, report, y_probs, y_pred



# Pipeline

class JobMatchingPipeline:
    def __init__(self, bucket_name, prefix="silver/"):
        self.s3_loader = S3DataLoader(bucket_name)
        self.embedding_processor = EmbeddingProcessor(self.s3_loader)
        self.similarity_calc = SimilarityCalculator()
        self.label_generator = LabelGenerator()
        self.bucket_name = bucket_name
        self.prefix = prefix

    def load_data(self):
        df_cvs = self.s3_loader.load_csv(self.prefix + "cv_filtred.csv")
        df_vagas = self.s3_loader.load_csv(self.prefix + "vagas_filtred.csv")
        df_cvs = df_cvs[["id", "cv_sugerido"]].rename(columns={"id": "cv_id"})
        df_vagas = df_vagas[["id", "titulo_vaga", "principais_atividades"]].rename(columns={"id": "job_id"})
        return df_cvs, df_vagas

    def run_pipeline(self, top_k=50):
        df_cvs, df_vagas = self.load_data()
        cv_embeddings = self.embedding_processor.get_cv_embeddings(df_cvs["cv_sugerido"].astype(str).tolist())

        pairs_df = self.similarity_calc.calculate_top_k_pairs(df_vagas, df_cvs, cv_embeddings, self.embedding_processor, top_k)
        pairs_df["tfidf_score"] = self.similarity_calc.calculate_tfidf_scores(pairs_df, df_vagas, df_cvs)
        pairs_df = self.similarity_calc.normalize_and_combine_scores(pairs_df)
        pairs_df["label"] = self.label_generator.create_top_percentile_labels(pairs_df)

        buffer = io.BytesIO()
        pairs_df.to_csv(buffer, index=False)
        self.s3_loader.upload_bytes(buffer.getvalue(), "silver/dataset_topk.csv")

        buffer = io.BytesIO()
        pairs_df.to_parquet(buffer, index=False)
        self.s3_loader.upload_bytes(buffer.getvalue(), "silver/dataset_topk.parquet")

        print(f"Pipeline concluída com {len(pairs_df)} pares")
        return pairs_df

    def train_model(self, pairs_df, epochs=10):
        model = MatchNet(input_dim=2)

        # calcular pos_weight para balancear loss
        n_pos = pairs_df["label"].sum()
        n_neg = len(pairs_df) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

        trainer = ModelTrainer(model, pos_weight=pos_weight)
        X_train, X_test, y_train, y_test = trainer.prepare_data(pairs_df)
        train_loader, test_loader = trainer.create_data_loaders(X_train, X_test, y_train, y_test)
        trainer.train(train_loader, epochs)
        acc, report, y_probs, y_pred = trainer.evaluate(X_test, y_test)
        print(f"Accuracy: {acc}")
        print(report)
        return model, trainer


if __name__ == "__main__":
    load_dotenv()
    pipeline = JobMatchingPipeline("bucket-tc5")
    pairs_df = pipeline.run_pipeline(top_k=50)
    model, trainer = pipeline.train_model(pairs_df, epochs=10)
