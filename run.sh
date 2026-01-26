#!/bin/bash


echo "🔍 Activating environment variables"
set .env
echo "🔍 Hosting on http://127.0.0.1:3000"
echo "📜 Swagger docs available at http://127.0.0.1:3000/docs"
echo "🛠️  Using: main:app with hot-reload enabled"

# Run the FastAPI app
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload