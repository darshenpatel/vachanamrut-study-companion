#!/bin/bash

# Vachanamrut Companion Backend Startup Script
# This script ensures the backend starts properly with all dependencies

set -e

echo "🚀 Starting Vachanamrut Companion Backend..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the backend directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found. Please set up the virtual environment first."
    echo "Run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python -c "import fastapi, uvicorn, anthropic, faiss" 2>/dev/null; then
    echo "⚠️  Some dependencies are missing. Installing from requirements.txt..."
    pip install -r requirements.txt
fi

# Start the backend server
echo "🌟 Starting FastAPI server..."
echo "📍 Backend will be available at: http://127.0.0.1:8000"
echo "📖 API Documentation at: http://127.0.0.1:8000/api/docs"
echo "💻 Press Ctrl+C to stop the server"
echo ""

# Run with uvicorn
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000