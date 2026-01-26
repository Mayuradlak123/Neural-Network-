# Neural Network API

A comprehensive FastAPI application for neural network modeling, ETL processes, and model interpretability.

## 🚀 Overview

This project provides a robust API for:
- **ANN & CNN Modeling**: Build and train Artificial and Convolutional Neural Networks.
- **ETL Processes**: Extract, Transform, and Load workflows for data processing.
- **Model Interpretability**: Tools for explaining model predictions (SHAP, LIME).
- **RAG Integration**: Retrieval-Augmented Generation using ChromaDB.
- **Async DB Connections**: Efficient MongoDB and ChromaDB integrations.

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Databases**: MongoDB, ChromaDB, Redis
- **ML libraries**: PyTorch, Scikit-learn, LangChain, NLTK, spaCy
- **Interpretability**: SHAP, LIME

## Docker Setup

The application is fully dockerized for easy deployment and development.

### Prerequisites
- Docker and Docker Compose installed.

### Quick Start
1.  **Configure Environment**: Ensure your `.env` file is present in the root directory.
2.  **Run with Docker Compose**:
    ```bash
    docker compose up --build
    ```

### Services
Once running, you can access the following services:
- **FastAPI Web App**: `http://localhost:8000`
- **FastAPI Docs (Swagger)**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **ChromaDB**: `http://localhost:8002`
- **MongoDB**: `mongodb://localhost:27017`
- **Redis**: `localhost:6379`

## Local Setup (Non-Docker)

### 1. Prerequisites
- Python 3.11 or higher.
- Local instances of MongoDB, ChromaDB, and Redis.

### 2. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the setup script if available
./setup.sh
```

### 3. Run the Server
```bash
./run.sh
# OR
uvicorn main:app --reload
```

---

## Project Structure
```text
Neural Network/
├── config/              # DB and Logger configurations
├── core/                # Core logic and utilities
├── routers/             # API and Web endpoints
├── services/            # Business logic and ML services
├── data/                # Raw and processed data
├── models/              # Trained model files
├── templates/           # HTML templates
├── main.py              # Application entry point
└── Dockerfile           # Multi-stage build config
```
