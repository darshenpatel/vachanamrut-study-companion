# Vachanamrut Study Companion - File Structure

## Root Directory Structure

```
vachanamrut-companion/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── frontend/
├── backend/
├── docs/
├── scripts/
└── data/
```

## Frontend Structure (React + TypeScript + Vite)

```
frontend/
├── package.json
├── package-lock.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── .env.example
├── public/
│   ├── favicon.ico
│   ├── manifest.json
│   └── icons/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── components/
│   │   ├── ui/
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Card.tsx
│   │   │   └── index.ts
│   │   ├── chat/
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── ChatInput.tsx
│   │   │   ├── TypingIndicator.tsx
│   │   │   └── index.ts
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Footer.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── index.ts
│   │   └── features/
│   │       ├── themes/
│   │       │   ├── ThemeExplorer.tsx
│   │       │   ├── ThemeCard.tsx
│   │       │   └── index.ts
│   │       └── citations/
│   │           ├── CitationDisplay.tsx
│   │           ├── ReferenceLink.tsx
│   │           └── index.ts
│   ├── hooks/
│   │   ├── useChat.ts
│   │   ├── useApi.ts
│   │   ├── useMobile.ts
│   │   └── index.ts
│   ├── services/
│   │   ├── api.ts
│   │   ├── chatService.ts
│   │   ├── themeService.ts
│   │   └── index.ts
│   ├── types/
│   │   ├── chat.ts
│   │   ├── api.ts
│   │   ├── theme.ts
│   │   └── index.ts
│   ├── utils/
│   │   ├── formatting.ts
│   │   ├── validation.ts
│   │   ├── constants.ts
│   │   └── index.ts
│   ├── context/
│   │   ├── ChatContext.tsx
│   │   ├── ThemeContext.tsx
│   │   └── index.ts
│   └── assets/
│       ├── images/
│       ├── icons/
│       └── fonts/
└── tests/
    ├── components/
    ├── hooks/
    ├── services/
    └── utils/
```

## Backend Structure (FastAPI + Python)

```
backend/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── themes.py
│   │   │   ├── health.py
│   │   │   └── search.py
│   │   └── dependencies.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai_service.py
│   │   ├── document_service.py
│   │   ├── theme_service.py
│   │   ├── vector_service.py
│   │   └── chat_service.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── document.py
│   │   ├── theme.py
│   │   └── response.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py
│   │   ├── embeddings.py
│   │   ├── retrieval.py
│   │   └── prompt_templates.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── theme.py
│   │   └── response.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── validation.py
│       ├── formatting.py
│       └── exceptions.py
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_api/
│   │   ├── __init__.py
│   │   ├── test_chat.py
│   │   ├── test_themes.py
│   │   └── test_search.py
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_ai_service.py
│   │   ├── test_document_service.py
│   │   └── test_theme_service.py
│   └── test_core/
│       ├── __init__.py
│       ├── test_pdf_processor.py
│       └── test_retrieval.py
├── scripts/
│   ├── setup_database.py
│   ├── process_documents.py
│   ├── load_theme_mappings.py
│   └── health_check.py
└── alembic/
    ├── env.py
    ├── script.py.mako
    └── versions/
```

## Data Directory

```
data/
├── raw/
│   ├── vachanamrut.pdf
│   └── theme_mappings.json
├── processed/
│   ├── documents/
│   │   ├── chunks/
│   │   └── metadata/
│   └── embeddings/
└── backup/
```

## Documentation Directory

```
docs/
├── README.md
├── api/
│   ├── endpoints.md
│   ├── authentication.md
│   └── examples.md
├── deployment/
│   ├── setup.md
│   ├── docker.md
│   └── production.md
├── development/
│   ├── getting-started.md
│   ├── testing.md
│   └── contributing.md
└── design/
    ├── architecture.md
    ├── database-schema.md
    └── ui-components.md
```

## Scripts Directory

```
scripts/
├── setup/
│   ├── install-dependencies.sh
│   ├── setup-environment.sh
│   └── initialize-database.sh
├── development/
│   ├── start-dev.sh
│   ├── run-tests.sh
│   └── lint-code.sh
├── deployment/
│   ├── build-docker.sh
│   ├── deploy-staging.sh
│   └── deploy-production.sh
└── data/
    ├── process-pdf.py
    ├── create-embeddings.py
    └── load-themes.py
```

## Key Configuration Files

### Root Level Files

**docker-compose.yml**
```yaml
# Development environment setup
# Frontend, backend, database services
# Volume mounts for development
```

**.env.example**
```env
# Environment variables template
# API keys, database URLs, service endpoints
```

**.gitignore**
```
# Standard ignores for Python, Node.js, environment files
# IDE files, OS files, logs, build artifacts
```

### Frontend Configuration

**package.json**
- React, TypeScript, Vite dependencies
- Development and build scripts
- Testing framework setup

**vite.config.ts**
- Development server configuration
- Build optimization settings
- Plugin configurations

**tailwind.config.js**
- Custom design system colors
- Mobile-first breakpoints
- Component-specific utilities

### Backend Configuration

**requirements.txt** / **pyproject.toml**
- FastAPI, LangChain, AI service dependencies
- Database drivers, testing frameworks
- Development tools and linting

**app/config.py**
- Environment-based configuration
- API keys and service endpoints
- Database connection settings

## Development Workflow Files

### Testing Configuration
- **pytest.ini** for backend testing
- **vitest.config.ts** for frontend testing
- Test data fixtures and mocks

### CI/CD Configuration
- **.github/workflows/** for GitHub Actions
- Build, test, and deployment pipelines
- Environment-specific configurations

### Code Quality
- **.eslintrc.js** for frontend linting
- **.pylintrc** for backend linting
- **prettier.config.js** for code formatting

This structure provides clear separation of concerns, makes it easy for Claude Code to understand the project organization, and follows best practices for both React and FastAPI applications.