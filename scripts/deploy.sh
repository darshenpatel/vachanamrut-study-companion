#!/bin/bash

# Deployment script for Vachanamrut Study Companion
set -e

echo "🙏 Deploying Vachanamrut Study Companion..."

# Check environment
ENVIRONMENT=${1:-development}
echo "📦 Environment: $ENVIRONMENT"

# Build frontend
echo "🔨 Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Test backend
echo "🧪 Testing backend system..."
cd backend
python3 test_lightweight_system.py
if [ $? -ne 0 ]; then
    echo "❌ Backend tests failed!"
    exit 1
fi
cd ..

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose build

if [ "$ENVIRONMENT" = "production" ]; then
    echo "🚀 Starting production deployment..."
    docker-compose --profile production up -d
else
    echo "🛠️  Starting development deployment..."
    docker-compose up -d
fi

# Health check
echo "🏥 Performing health check..."
sleep 10

# Check backend health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
    exit 1
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is accessible"
else
    echo "❌ Frontend health check failed"
    exit 1
fi

echo ""
echo "🎉 Deployment successful!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/api/docs"
echo ""
echo "💡 To test the system:"
echo "   curl http://localhost:8000/health"
echo "   python3 backend/test_lightweight_system.py"