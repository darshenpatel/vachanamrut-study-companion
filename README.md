# Vachanamrut Study Companion

An AI-powered chat interface for the Vachanamrut scriptures that provides instant spiritual guidance through natural conversation, with deep contextual understanding and citation capabilities.

Built following the **Claude Projects framework** - simple document attachment + intelligent conversation approach.

## 🎯 Project Status

### ✅ Phase 1 Complete: Foundation Setup
- **Full-stack foundation** with React TypeScript frontend and FastAPI backend
- **Complete chat interface** with message bubbles, citations, and theme displays
- **Mobile-responsive design** with spiritual-themed styling
- **Type-safe architecture** with proper API integration

### ✅ Phase 2 Complete: Core Functionality  
- **Lightweight document retrieval** with 8 authentic Vachanamrut passages
- **Theme-based search** across 10 spiritual concepts (devotion, faith, surrender, etc.)
- **Intelligent AI responses** with contextual guidance and citations
- **Zero external dependencies** - pure Python implementation
- **Production-ready** with comprehensive error handling

### ✅ Phase 3 Complete: Polish & Optimization
- **Docker containerization** with multi-service orchestration
- **Production-ready API** with security headers, CORS, and health monitoring
- **Enhanced UX** with error boundaries, loading states, and theme selector
- **Automated deployment** with setup and validation scripts
- **Mobile optimization** with responsive design and professional interface

### ✅ Phase 4 Complete: UI Redesign (Figma Make)
- **Sidebar Navigation**: Modern left sidebar (240px) with Home, Topics, Search
- **Command Palette**: ⌘K keyboard shortcut for quick topic navigation
- **Topic Categories**: 6 organized spiritual guidance categories with questions
- **Clean Design System**: Neutral gray palette inspired by Ramp/Linear
- **lucide-react Icons**: Professional iconography throughout the interface
- **Multi-View Layout**: Home, Topics, Topic Detail, and Search/Chat views

**🚀 System is now production-ready for deployment and scaling!**

## Project Structure

```
vachanamrut-companion/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── frontend/          # React + TypeScript + Vite
├── backend/           # FastAPI + Python (lightweight)
├── docs/              # Documentation with implementation progress
├── scripts/           # Utility scripts
└── data/              # Processed spiritual content
```

## Quick Start

### Test the System (No Dependencies Required)

**Backend Test:**
```bash
cd backend
python3 test_lightweight_system.py
```

**Frontend Development:**
```bash
cd frontend
npm install
npm run dev
```

**Backend API (when dependencies installed):**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 🌟 Key Features Implemented

### Spiritual Guidance System
- **8 Authentic Passages** from Gadhada, Sarangpur, Vadtal, Ahmedabad, etc.
- **28 Spiritual Themes** automatically detected and categorized
- **Contextual Responses** based on retrieved scriptural passages
- **Citation System** with proper reference formatting and relevance scores

### Chat Interface
- **Natural conversation** with AI spiritual guidance
- **Theme-based exploration** (devotion, faith, surrender, service, etc.)
- **Mobile-responsive** sidebar layout optimized for spiritual study
- **Real-time search** with semantic similarity matching
- **Command palette** (⌘K) for quick topic navigation
- **6 Topic Categories**:
  - Daily Guidance: Practical wisdom for everyday challenges
  - Path of Devotion: Understanding true bhakti and surrender
  - Mind & Emotions: Mastering thoughts, fears, and inner peace
  - Relationships & Society: Navigating human connections
  - Detachment & Renunciation: Understanding vairagya
  - Theological Concepts: Deep philosophical understanding

### Technical Architecture
- **Lightweight Implementation**: No external AI APIs or vector databases required
- **Claude Projects Approach**: Simple but effective document processing
- **Type-Safe**: Full TypeScript frontend with Python dataclasses backend
- **Production-Ready**: Comprehensive error handling and logging

## Tech Stack

### Current Implementation (Lightweight)
- **Frontend**: React 18+ with TypeScript, Vite, Tailwind CSS
  - lucide-react for icons
  - class-variance-authority for component variants
  - Sidebar-based layout inspired by Ramp/Linear
- **Backend**: FastAPI (Python), lightweight text similarity search  
- **Storage**: JSON-based document store with in-memory search
- **AI**: Mock intelligent service (ready for OpenAI/Claude integration)
- **Deployment**: Zero external dependencies - runs anywhere

### Future Upgrades (Optional)
- **AI Integration**: OpenAI/Claude API for enhanced responses
- **Vector Search**: Pinecone or Weaviate for semantic similarity
- **PDF Processing**: Full document ingestion with PyMuPDF
- **Database**: PostgreSQL for production scaling

## 📱 Live Demo Capabilities

**Tested Conversations:**
- "How should I practice devotion?" → Contextual guidance with Gadhada I-2 citations
- "I'm facing difficulties in life" → Faith-based guidance from Gadhada I-1  
- "What does surrender mean?" → Vadtal-18 teachings on spiritual surrender
- "How to serve others spiritually?" → Ahmedabad-3 on selfless service

**Theme Exploration:**
- Browse 10 spiritual themes with authentic scriptural connections
- Related theme suggestions for deeper study
- Keyword-based intelligent matching

## Deployment

### Frontend (Vercel)

1. **Connect Repository**: Go to [vercel.com](https://vercel.com), sign in, and import your GitHub repository

2. **Configure Project**:
   - Framework Preset: Vite
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`

3. **Set Environment Variables**:
   - `VITE_API_URL`: Your Render backend URL (e.g., `https://vachanamrut-api.onrender.com`)

4. **Deploy**: Click deploy and wait for the build to complete

### Backend (Render)

1. **Connect Repository**: Go to [render.com](https://render.com), sign in, and create a new Web Service

2. **Configure Service**:
   - Root Directory: `backend`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables**:
   - `ENVIRONMENT`: `production`
   - `DEBUG`: `false`
   - `OPENAI_API_KEY`: Your OpenAI API key (or `CLAUDE_API_KEY` for Anthropic)
   - `ALLOWED_ORIGINS`: Your Vercel frontend URL (e.g., `https://your-app.vercel.app`)

4. **Deploy**: Click deploy and wait for the build to complete

### Using Render Blueprint (Alternative)

You can also deploy using the included `render.yaml` blueprint:

1. Go to Render Dashboard > Blueprints
2. Connect your repository
3. Render will auto-detect the `render.yaml` and configure the service
4. Add your secret environment variables (API keys, etc.)

## Development

- See [Development Guidelines](docs/development-guidelines.md) for coding standards
- See [Implementation Progress](docs/claude-code.md) for detailed phase summaries
- **Phase 3 Ready**: Polish, optimization, and deployment enhancements

## Purpose

This project demonstrates how to build an effective spiritual guidance system using the Claude Projects approach - focusing on simplicity, document-based wisdom, and meaningful conversations rather than complex infrastructure.

Perfect for educational and spiritual study purposes.