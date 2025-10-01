# 🎯 Job Matching

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-ee4c2c.svg)
![Django](https://img.shields.io/badge/Django-5.2+-092e20.svg)
![Docker](https://img.shields.io/badge/Docker-28.4+-2496ed.svg)
![AWS S3](https://img.shields.io/badge/AWS-S3-orange.svg)
![Pentaho](https://img.shields.io/badge/Pentaho-9.4-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194e2.svg)
![Prometheus](https://img.shields.io/badge/Prometheus-2.40+-e6522c.svg)
![Grafana](https://img.shields.io/badge/Grafana-9.0+-f46800.svg)
![License](https://img.shields.io/badge/Google-cloud-white.svg)

Sistema inteligente de matching entre vagas e currículos usando Machine Learning, desenvolvido com Django, PyTorch e BERT embeddings.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Tecnologias](#tecnologias)
- [Arquitetura](#arquitetura)
- [Funcionalidades](#funcionalidades)
- [Instalação](#instalação)
- [Uso da API](#uso-da-api)
- [Monitoramento](#monitoramento)
- [Deploy](#deploy)

## 🚀 Sobre o Projeto

O Job Matching System é uma solução de Machine Learning que utiliza modelos de linguagem natural (BERT) para encontrar os melhores candidatos para cada vaga de emprego. O sistema combina embeddings semânticos com análise TF-IDF para criar um score final de compatibilidade.

### Principais Características

- **Embeddings Semânticos**: Utiliza o modelo `all-MiniLM-L6-v2` para representação vetorial de textos
- **Scoring Híbrido**: Combina similaridade semântica (70%) e TF-IDF (30%)
- **Rede Neural**: Modelo personalizado (MatchNet) para refinamento de predições
- **Rastreamento MLflow**: Tracking completo de experimentos e métricas
- **Monitoramento Prometheus/Grafana**: Observabilidade em tempo real
- **Armazenamento S3**: Dados e artefatos salvos na AWS

## 🛠 Tecnologias

### Backend & ML
- Python 3.x
- Django + Django REST Framework
- PyTorch
- Sentence Transformers (BERT)
- Scikit-learn
- MLflow

### ETL & Data Processing
- Pentaho Data Integration 9.4
- Pandas
- NumPy

### Infraestrutura & Monitoramento
- Docker & Docker Compose
- AWS S3
- Google Cloude
- Prometheus
- Grafana
- PostgreSQL

### APIs & Documentação
- Swagger/OpenAPI (drf-yasg)
- ReDoc

## 💻 Componentes Principais

### 1. **EmbeddingProcessor**
Gerencia embeddings BERT para currículos e vagas
- Cache automático em S3
- Suporte GPU/CPU
- Lazy loading do modelo

### 2. **SimilarityCalculator**
Calcula múltiplas métricas de similaridade
- Semantic Search (embeddings)
- TF-IDF cosine similarity
- Score normalizado e combinado

### 3. **MatchNet (PyTorch)**
Rede neural para classificação binária
- Arquitetura: [2 → 16 → 8 → 1]
- Loss: BCEWithLogitsLoss (com class weight)
- Optimizer: Adam

### 4. **JobMatchingPipeline**
Orquestra todo o fluxo de ML
- Carregamento de dados
- Feature engineering
- Treinamento e avaliação
- Logging MLflow + Prometheus

## ✨ Funcionalidades

### API Endpoints

#### Vagas e Currículos
```
GET  /api/vagas/                  # Lista todas as vagas
GET  /api/vagas/{id}/             # Detalhes de uma vaga
GET  /api/vagas/{id}/predict/     # Top-K candidatos para a vaga
GET  /api/curriculos/             # Lista todos os currículos
GET  /api/curriculos/{id}/        # Detalhes de um currículo
```

#### Machine Learning
```
POST /api/ml/load/                # Carrega embeddings e dados do S3
POST /api/ml/train/               # Treina o modelo MatchNet
GET  /api/ml/metrics/             # Retorna métricas do último treino
GET  /api/ml/health/              # Status do sistema ML
```

#### Documentação
```
GET  /swagger/                    # Interface Swagger UI
GET  /redoc/                      # Interface ReDoc
```

## 🔧 Instalação

### Pré-requisitos

- Docker & Docker Compose
- Credenciais AWS (S3)
- Python 3.12+

### Configuração

1. **Clone o repositório**
```bash
git clone <seu-repositorio>
cd job-matching
```

2. **Configure variáveis de ambiente**

Crie um arquivo `.env` na raiz do projeto:

```env
# AWS
AWS_ACCESS_KEY_ID=sua_access_key
AWS_SECRET_ACCESS_KEY=sua_secret_key
AWS_DEFAULT_REGION=us-east-1

# Django
DEBUG=1
SECRET_KEY=sua_secret_key_django

# Database (opcional)
POSTGRES_DB=tc5_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

3. **Inicie os containers**
```bash
docker-compose up -d
```

4. **Acesse os serviços**
- API: http://localhost:8000
- Swagger: http://localhost:8000/swagger/
- MLflow: http://localhost:5001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/123456)

## 📖 Uso da API

### 1. Carregar Artefatos

Primeiro, carregue os embeddings e dados do S3:

```bash
curl -X POST http://localhost:8000/api/ml/load/
```

**Response:**
```json
{
  "message": "Artefatos carregados com sucesso",
  "curriculos": 1500,
  "vagas": 200,
  "embeddings_shape": [1500, 384]
}
```

### 2. Obter Candidatos para uma Vaga

```bash
curl "http://localhost:8000/api/vagas/42/predict/?top_k=5"
```

**Response:**
```json
{
  "vaga_id": "42",
  "vaga_titulo": "Desenvolvedor Python Senior",
  "candidatos": [
    {
      "cv_id": 789,
      "texto": "Desenvolvedor Python com 5 anos...",
      "score": 0.8542,
      "match_percentage": 85.42
    }
  ],
  "total_candidatos": 5
}
```

### 3. Treinar Modelo

```bash
curl -X POST http://localhost:8000/api/ml/train/ \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "test_size": 0.2,
    "top_k": 50
  }'
```

**Response:**
```json
{
  "message": "Modelo treinado com sucesso",
  "metrics": {
    "accuracy": 0.8723,
    "avg_precision_score": 0.7891,
    "macro_avg_f1_score": 0.8156,
    "mlflow_run_id": "abc123def456"
  }
}
```

### 4. Verificar Saúde do Sistema

```bash
curl http://localhost:8000/api/ml/health/
```

## 📊 Monitoramento

### Prometheus Metrics

O sistema expõe métricas na porta 8001:

```
http://localhost:8001/metrics
```

**Métricas Disponíveis:**
- `ml_predictions_total` - Total de predições por versão
- `ml_prediction_score` - Distribuição de scores
- `ml_training_accuracy` - Acurácia do modelo
- `ml_training_loss` - Loss durante treinamento

### MLflow Tracking

Acesse o MLflow UI para visualizar:
- Histórico de experimentos
- Comparação de hiperparâmetros
- Métricas de treino/teste
- Curvas Precision-Recall
- Matrizes de confusão

```
http://localhost:5001
```

### Grafana Dashboards

Dashboards pré-configurados em:
```
http://localhost:3001
```

Credenciais padrão: `admin` / `123456`

## 🌐 Deploy

### Projeto em Produção

O projeto está hospedado no **Google Cloud Platform**.<br>
Teste a API e explore as funcionalidades diretamente no navegador!

**🌐 Acesse a aplicação:** http://34.133.21.83:8000/

**📚 Base URL:** `http://34.133.21.83:8000/api/`

### Endpoints Públicos

```
# Documentação
http://34.133.21.83:8000/swagger/
http://34.133.21.83:8000/redoc/

# API
http://34.133.21.83:8000/api/vagas/
http://34.133.21.83:8000/api/curriculos/
http://34.133.21.83:8000/api/ml/health/
```

### Exemplo de Uso (Produção)

```bash
# Listar vagas
curl http://34.133.21.83:8000/api/vagas/

# Predição para vaga específica
curl "http://34.133.21.83:8000/api/vagas/1/predict/?top_k=10"

# Status do sistema
curl http://34.133.21.83:8000/api/ml/health/
```

## 📁 Estrutura do Projeto

```
.
├── ML/
│   ├── ML_bert.py              # Pipeline principal de ML
│   ├── metrics.py              # Métricas Prometheus
│   └── monitoring.py           # Logging S3
├── api/
│   ├── models.py               # Models Django
│   ├── serializers.py          # DRF Serializers
│   ├── views.py                # API Views
│   └── urls.py                 # Rotas
├── templates/
│   └── vagas.html              # Frontend
├── grafana/
│   ├── dashboards/             # Dashboards JSON
│   └── provisioning/           # Datasources
├── docker-compose.yml          # Orquestração
├── Dockerfile                  # Imagem Python
├── prometheus.yml              # Config Prometheus
└── requirements.txt            # Dependências Python
```

## 👥 Autores

Desenvolvido para o Tech Challenge 5 - FIAP
