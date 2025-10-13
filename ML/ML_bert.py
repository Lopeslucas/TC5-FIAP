#ML/ML_bert.py

# Importações necessárias para o projeto
import os
import io
import boto3  # Para conectar com a AWS S3
import torch  # Framework de deep learning
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

# Importações do MLflow
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import tempfile
import joblib

# Importações prometheus
from ML.monitoring import log_predictions_to_s3, log_training_summary
from ML.metrics import log_prediction, log_training_metrics

# Classe responsável por carregar e enviar dados do/para o S3 (armazenamento AWS)
class S3DataLoader:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        # Cria conexão com S3 usando credenciais do arquivo .env
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )

    # Carrega um arquivo CSV do S3 e retorna como DataFrame
    def load_csv(self, key):
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        return pd.read_csv(obj["Body"], sep=";")

    # Faz upload de dados (em bytes) para o S3
    def upload_bytes(self, data: bytes, s3_path: str):
        self.s3.upload_fileobj(io.BytesIO(data), self.bucket_name, s3_path)
        print(f"Upload concluído: s3://{self.bucket_name}/{s3_path}")


# Classe que processa textos e cria embeddings (representações numéricas de textos)
class EmbeddingProcessor:
    def __init__(self, s3_loader: S3DataLoader, model_name="all-MiniLM-L6-v2"):
        # Define se vai usar GPU (cuda) ou CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None  # Modelo só é carregado quando necessário
        self.s3_loader = s3_loader

    # Garante que o modelo está carregado na memória
    def _ensure_model_loaded(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    # Busca embeddings de CVs do S3 ou cria novos se não existirem
    def get_cv_embeddings(self, cv_texts, s3_path="gold/cv_embeddings.npy", dtype=np.float32):
        try:
            # Tenta carregar embeddings já salvos no S3
            obj = self.s3_loader.s3.get_object(Bucket=self.s3_loader.bucket_name, Key=s3_path)
            array = np.load(io.BytesIO(obj["Body"].read()))
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        except self.s3_loader.s3.exceptions.NoSuchKey:
            # Se não existir, cria novos embeddings
            self._ensure_model_loaded()
            embeddings = self.model.encode(cv_texts, convert_to_tensor=True).to(torch.float32)
            # Salva os embeddings no S3 para uso futuro
            buffer = io.BytesIO()
            np.save(buffer, embeddings.cpu().numpy().astype(dtype))
            buffer.seek(0)
            self.s3_loader.upload_bytes(buffer.read(), s3_path)
            return embeddings

    # Converte um texto em embedding
    def encode_text(self, text):
        self._ensure_model_loaded()
        return self.model.encode(text, convert_to_tensor=True)


# Classe que calcula a similaridade entre vagas e CVs
class SimilarityCalculator:
    def __init__(self):
        # TF-IDF: técnica para medir importância de palavras em documentos
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        # MinMaxScaler: normaliza valores entre 0 e 1
        self.scaler = MinMaxScaler()

    # Encontra os top K CVs mais similares para cada vaga usando embeddings
    def calculate_top_k_pairs(self, df_vagas, df_cvs, cv_embeddings, embedding_processor, top_k=50):
        # Combina título e atividades das vagas em um texto único
        vaga_texts = (df_vagas["titulo_vaga"].astype(str) + " " + df_vagas["principais_atividades"].astype(str)).tolist()
        
        # Cria embeddings para as vagas
        embedding_processor._ensure_model_loaded()
        vaga_embeddings = embedding_processor.model.encode(vaga_texts, convert_to_tensor=True)
        vaga_embeddings = vaga_embeddings.to(cv_embeddings.dtype)

        # Busca os CVs mais similares para cada vaga
        hits = util.semantic_search(vaga_embeddings, cv_embeddings, top_k=top_k)
        all_pairs = []

        # Cria pares (vaga, CV) com score de similaridade
        for i, job_id in enumerate(df_vagas["job_id"]):
            for h in hits[i]:
                all_pairs.append({
                    "job_id": job_id,
                    "cv_id": df_cvs.iloc[h["corpus_id"]]["cv_id"],
                    "semantic_score": float(h["score"])  # Score baseado em embeddings
                })

        return pd.DataFrame(all_pairs)

    # Calcula similaridade usando TF-IDF (método complementar aos embeddings)
    def calculate_tfidf_scores(self, pairs_df, df_vagas, df_cvs):
        # Prepara textos de vagas e CVs
        vaga_texts = (df_vagas["titulo_vaga"].astype(str) + " " + df_vagas["principais_atividades"].astype(str)).tolist()
        cv_texts = df_cvs["cv_sugerido"].astype(str).tolist()
        all_texts = vaga_texts + cv_texts

        # Cria matriz TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        vaga_tfidf = tfidf_matrix[:len(vaga_texts)]
        cv_tfidf = tfidf_matrix[len(vaga_texts):]

        # Calcula similaridade de cosseno entre vagas e CVs
        scores_matrix = cosine_similarity(vaga_tfidf, cv_tfidf)
        # Extrai score para cada par vaga-CV
        tfidf_scores = [scores_matrix[df_vagas.index[df_vagas["job_id"] == row.job_id][0],
                                        df_cvs.index[df_cvs["cv_id"] == row.cv_id][0]]
                        for _, row in pairs_df.iterrows()]
        return tfidf_scores

    # Normaliza scores e combina similaridade semântica + TF-IDF em um score final
    def normalize_and_combine_scores(self, pairs_df, semantic_weight=0.7, tfidf_weight=0.3):
        # Normaliza ambos os scores entre 0 e 1
        pairs_df[["semantic_norm", "tfidf_norm"]] = self.scaler.fit_transform(
            pairs_df[["semantic_score", "tfidf_score"]]
        )
        # Combina os scores com pesos (70% semântico, 30% TF-IDF)
        pairs_df["final_score"] = semantic_weight * pairs_df["semantic_norm"] + tfidf_weight * pairs_df["tfidf_norm"]
        return pairs_df


# Classe que gera labels (rótulos) para treinar o modelo
class LabelGenerator:
    @staticmethod
    def create_top_percentile_labels(pairs_df, percentile=0.9):
        # Para cada vaga, marca como 1 (match) os CVs no top 10% de score
        labels = []
        for _, group in pairs_df.groupby("job_id"):
            threshold = group["final_score"].quantile(percentile)  # Calcula o percentil 90
            labels.extend((group["final_score"] >= threshold).astype(int).tolist())
        return labels


# Rede neural que aprende a prever se uma vaga combina com um CV
class MatchNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        # Cria camadas ocultas da rede neural
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])  # Camada linear + ativação ReLU
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Camada de saída (1 valor: probabilidade de match)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Classe responsável por treinar o modelo de rede neural
class ModelTrainer:
    def __init__(self, model, pos_weight=None, lr=0.001):
        # Define função de perda (loss) com ou sem balanceamento de classes
        if pos_weight is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Dá mais peso à classe positiva
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)  # Otimizador Adam
        self.lr = lr
        self.pos_weight = pos_weight

    # Prepara dados dividindo em treino e teste, respeitando grupos (job_ids)
    def prepare_data(self, pairs_df, features=["semantic_norm", "tfidf_norm"], test_size=0.2):
        X = pairs_df[features].values.astype("float32")  # Features (características)
        y = pairs_df["label"].values.astype("float32")  # Labels (0 ou 1)
        groups = pairs_df["job_id"].values  # Grupos para não misturar vagas entre treino/teste

        # Divide dados mantendo vagas inteiras em treino ou teste
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))

        self.train_idx = train_idx
        self.test_idx = test_idx

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    # Cria DataLoaders para processar dados em batches durante o treinamento
    def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=64):
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        return train_loader, test_loader

    # Treina o modelo por várias épocas
    def train(self, train_loader, epochs=10):
        train_losses = []
        
        for epoch in range(epochs):
            self.model.train()  # Coloca modelo em modo treino
            total_loss = 0.0
            # Processa cada batch de dados
            for xb, yb in train_loader:
                self.optimizer.zero_grad()  # Zera gradientes
                outputs = self.model(xb).squeeze(1)  # Faz predição
                loss = self.criterion(outputs, yb)  # Calcula erro
                loss.backward()  # Calcula gradientes
                self.optimizer.step()  # Atualiza pesos
                total_loss += loss.item() * xb.size(0)
            
            epoch_loss = total_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            # Registra loss no MLflow para acompanhamento
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return train_losses

    # Avalia performance do modelo nos dados de teste
    def evaluate(self, X_test, y_test):
        self.model.eval()  # Coloca modelo em modo avaliação
        with torch.no_grad():  # Não calcula gradientes (mais rápido)
            logits = self.model(torch.tensor(X_test)).squeeze(1)
            y_probs = torch.sigmoid(logits).numpy()  # Converte para probabilidades
            y_pred = (y_probs >= 0.5).astype(int)  # Classifica: 0 ou 1

        # Calcula métricas de performance
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Cria curva precision-recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
        avg_prec = average_precision_score(y_test, y_probs)

        # Plota e salva gráfico da curva PR no MLflow
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"AP={avg_prec:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(tmp_file.name, "plots")
        plt.close()

        # Cria e salva matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Adiciona valores numéricos na matriz
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


# Pipeline completo que orquestra todo o processo de matching de vagas
class JobMatchingPipeline:
    def __init__(self, bucket_name, prefix="silver/", mlflow_experiment_name="job-matching-v2"):
        self.s3_loader = S3DataLoader(bucket_name)
        self.embedding_processor = EmbeddingProcessor(self.s3_loader)
        self.similarity_calc = SimilarityCalculator()
        self.label_generator = LabelGenerator()
        self.bucket_name = bucket_name
        self.prefix = prefix
        
        # Configura experimento no MLflow
        mlflow.set_experiment(mlflow_experiment_name)
        
    # Carrega dados de CVs e vagas do S3
    def load_data(self):
        df_cvs = self.s3_loader.load_csv(self.prefix + "cv_filtred.csv")
        df_vagas = self.s3_loader.load_csv(self.prefix + "vagas_filtred.csv")
        # Renomeia colunas para padronizar
        df_cvs = df_cvs[["id", "cv_sugerido"]].rename(columns={"id": "cv_id"})
        df_vagas = df_vagas[["id", "titulo_vaga", "principais_atividades"]].rename(columns={"id": "job_id"})
        return df_cvs, df_vagas

    # Executa pipeline completo: carrega dados, calcula similaridades, gera labels
    def run_pipeline(self, top_k=50):
        df_cvs, df_vagas = self.load_data()
        
        # Registra parâmetros no MLflow
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("cv_count", len(df_cvs))
        mlflow.log_param("vaga_count", len(df_vagas))
        mlflow.log_param("embedding_model", self.embedding_processor.model_name)
        
        # Obtém embeddings dos CVs
        cv_embeddings = self.embedding_processor.get_cv_embeddings(df_cvs["cv_sugerido"].astype(str).tolist())

        # Calcula pares vaga-CV com maiores similaridades
        pairs_df = self.similarity_calc.calculate_top_k_pairs(df_vagas, df_cvs, cv_embeddings, self.embedding_processor, top_k)
        # Adiciona score TF-IDF
        pairs_df["tfidf_score"] = self.similarity_calc.calculate_tfidf_scores(pairs_df, df_vagas, df_cvs)
        # Normaliza e combina scores
        pairs_df = self.similarity_calc.normalize_and_combine_scores(pairs_df)
        # Gera labels para treino
        pairs_df["label"] = self.label_generator.create_top_percentile_labels(pairs_df)

        # Registra estatísticas dos dados
        mlflow.log_metric("total_pairs", len(pairs_df))
        mlflow.log_metric("positive_pairs", pairs_df["label"].sum())
        mlflow.log_metric("negative_pairs", len(pairs_df) - pairs_df["label"].sum())
        mlflow.log_metric("positive_ratio", pairs_df["label"].mean())

        # Salva dataset em CSV no S3
        buffer = io.BytesIO()
        pairs_df.to_csv(buffer, index=False)
        self.s3_loader.upload_bytes(buffer.getvalue(), "silver/dataset_topk.csv")

        # Salva dataset em Parquet no S3 (formato mais eficiente)
        buffer = io.BytesIO()
        pairs_df.to_parquet(buffer, index=False)
        self.s3_loader.upload_bytes(buffer.getvalue(), "silver/dataset_topk.parquet")

        # Salva dataset como artefato no MLflow
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            pairs_df.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, "datasets")

        print(f"Pipeline concluída com {len(pairs_df)} pares")
        return pairs_df

    # Treina modelo de rede neural para prever matches
    def train_model(self, pairs_df, epochs=10, test_size=0.2, batch_size=64, lr=0.001, 
                   semantic_weight=0.7, tfidf_weight=0.3, hidden_dims=[16, 8], seed=42):
            
        # Define seeds para reprodutibilidade
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Finaliza qualquer run MLflow ativa
        try:
            mlflow.end_run()
        except:
            pass
        
        with mlflow.start_run() as run:
            
            # Registra todos os hiperparâmetros no MLflow
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
            
            # Cria modelo
            model = MatchNet(input_dim=2, hidden_dims=hidden_dims)

            # Calcula peso para balancear classes (há mais negativos que positivos)
            n_pos = pairs_df["label"].sum()
            n_neg = len(pairs_df) - n_pos
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
            
            mlflow.log_param("pos_weight", pos_weight.item())

            # Cria trainer e prepara dados
            trainer = ModelTrainer(model, pos_weight=pos_weight, lr=lr)
            X_train, X_test, y_train, y_test = trainer.prepare_data(pairs_df, test_size=test_size)
            train_loader, test_loader = trainer.create_data_loaders(X_train, X_test, y_train, y_test, batch_size=batch_size)
            
            # Registra tamanhos dos conjuntos
            mlflow.log_metrics({
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum())
            })
            
            # Treina modelo
            print("Iniciando treinamento...")
            train_losses = trainer.train(train_loader, epochs)

            # Avalia modelo
            print("Avaliando modelo...")
            acc, report_dict, y_probs, y_pred = trainer.evaluate(X_test, y_test)
            avg_prec = average_precision_score(y_test, y_probs)
            cm = confusion_matrix(y_test, y_pred).tolist()

            # Extrai chaves das classes do relatório (pode ser string ou int)
            class_0_key = "0" if "0" in report_dict else 0
            class_1_key = "1" if "1" in report_dict else 1
            
            # Prepara métricas principais
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
            
            # Adiciona métricas específicas de cada classe
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
            
            # Registra métricas no MLflow
            mlflow.log_metrics(metrics)

            # Salva modelo treinado no MLflow
            signature = infer_signature(X_test, y_probs)
            mlflow.pytorch.log_model(
                model, 
                "model",
                signature=signature,
                input_example=X_test[:5] if len(X_test) > 5 else X_test
            )

            # Salva modelo no S3 também
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                torch.save(model.state_dict(), tmp_file.name)
                with open(tmp_file.name, 'rb') as f:
                    self.s3_loader.upload_bytes(f.read(), "gold/model.pth")
                        
            # Salva scaler (normalizador) como artefato
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                joblib.dump(self.similarity_calc.scaler, tmp_file.name)
                mlflow.log_artifact(tmp_file.name, "preprocessors/scaler.pkl")
                
            # Salva vectorizer TF-IDF como artefato
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                joblib.dump(self.similarity_calc.tfidf_vectorizer, tmp_file.name)
                mlflow.log_artifact(tmp_file.name, "preprocessors/tfidf_vectorizer.pkl")

            # Prepara métricas completas para salvar
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

            # Salva métricas em JSON no S3
            buffer = io.BytesIO()
            buffer.write(json.dumps(full_metrics, indent=2).encode("utf-8"))
            buffer.seek(0)
            self.s3_loader.upload_bytes(buffer.read(), "gold/metrics.json")
            
            # Salva métricas também no MLflow
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                with open(tmp_file.name, 'w') as f:
                    json.dump(full_metrics, f, indent=2)
                mlflow.log_artifact(tmp_file.name, "metrics/full_metrics.json")

            # Adiciona tags ao experimento MLflow
            mlflow.set_tags({
                "model_type": "job_matching",
                "framework": "pytorch",
                "data_source": "s3",
                "bucket": self.bucket_name
            })

            # Imprime resumo dos resultados
            print(f"Treinamento concluído!")
            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Average Precision: {avg_prec:.4f}")

            # Imprime resumo dos resultados
            print(f"Treinamento concluído!")
            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Average Precision: {avg_prec:.4f}")

            # ====== MONITORAMENTO SIMPLIFICADO ======
            test_pairs = pairs_df.iloc[trainer.test_idx] if hasattr(trainer, 'test_idx') else pairs_df.tail(len(X_test))
            
            # Salva predições no S3
            log_predictions_to_s3(
                pairs_df=test_pairs.reset_index(drop=True),
                y_probs=y_probs,
                model_version=run.info.run_id,
                s3_loader=self.s3_loader
            )
            
            # Registra métricas no Prometheus
            log_training_metrics(
                accuracy=acc,
                loss=train_losses[-1] if train_losses else 0.0,
                model_version=run.info.run_id
            )
            
            # Registra cada predição
            for score in y_probs:
                log_prediction(score, model_version=run.info.run_id)
            
            # Salva resumo completo no S3
            log_training_summary(
                metrics=full_metrics,
                model_version=run.info.run_id,
                s3_loader=self.s3_loader
            )

            return model, trainer, full_metrics


# Execução principal do script
if __name__ == "__main__":
    load_dotenv()  # Carrega variáveis de ambiente do arquivo .env
    
    # Cria pipeline de matching de vagas
    pipeline = JobMatchingPipeline("bucket-tc5", mlflow_experiment_name="job-matching-experiment")
    # Executa pipeline para gerar dataset
    pairs_df = pipeline.run_pipeline(top_k=50)
    # Treina modelo com os dados gerados
    model, trainer, metrics = pipeline.train_model(
        pairs_df, 
        epochs=10,
        lr=0.001,
        batch_size=64,
        hidden_dims=[16, 8]
    )