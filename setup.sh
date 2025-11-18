#!/bin/bash

echo "🔧 Setting up Python environment for Video Transcription AI..."


# Step 2: Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Step 3: Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
echo "📦 Installing packages from requirements.txt..."
pip install -r requirements.txt

# Step 5: Done
echo "✅ Setup complete."
echo "🚀 To run the server: source venv/bin/activate && uvicorn main:app --reload"
    
set .env