#!/bin/bash

# Setup script for Vachanamrut Study Companion
set -e

echo "🙏 Setting up Vachanamrut Study Companion..."

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required. Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/v//')
echo "✅ Node.js version: $NODE_VERSION"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install Python 3.11+ from https://python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | sed 's/Python //')
echo "✅ Python version: $PYTHON_VERSION"

# Check Docker (optional)
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | sed 's/Docker version //' | sed 's/,.*//')
    echo "✅ Docker version: $DOCKER_VERSION"
else
    echo "⚠️  Docker not found (optional for development)"
fi

# Setup frontend
echo ""
echo "🎨 Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
else
    echo "✅ Frontend dependencies already installed"
fi

echo "🔨 Building frontend..."
npm run build

cd ..

# Setup backend
echo ""
echo "⚙️  Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "🔗 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing backend dependencies..."
pip install --upgrade pip
# Install minimal dependencies for the lightweight system
pip install fastapi uvicorn python-dotenv

echo "🧪 Testing backend system..."
python3 test_lightweight_system.py

if [ $? -eq 0 ]; then
    echo "✅ Backend system test passed!"
else
    echo "❌ Backend system test failed!"
    exit 1
fi

cd ..

# Create data directories
echo ""
echo "📁 Setting up data directories..."
mkdir -p data/processed data/raw data/backup
echo "✅ Data directories created"

# Copy environment files
echo ""
echo "⚙️  Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Environment file created (.env)"
    echo "💡 Please edit .env file to configure API keys if needed"
else
    echo "✅ Environment file already exists"
fi

if [ ! -f "frontend/.env" ]; then
    echo "VITE_API_URL=http://localhost:8000" > frontend/.env
    echo "✅ Frontend environment file created"
fi

if [ ! -f "backend/.env" ]; then
    cp backend/.env.example backend/.env
    echo "✅ Backend environment file created"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start the application:"
echo ""
echo "   Option 1 - Docker (recommended):"
echo "   docker-compose up -d"
echo ""
echo "   Option 2 - Manual:"
echo "   # Terminal 1 (Backend):"
echo "   cd backend && source venv/bin/activate && python3 -m uvicorn app.main:app --reload"
echo ""
echo "   # Terminal 2 (Frontend):"
echo "   cd frontend && npm run dev"
echo ""
echo "🌐 Application URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/api/docs"
echo ""
echo "🧪 To test the system:"
echo "   cd backend && python3 test_lightweight_system.py"