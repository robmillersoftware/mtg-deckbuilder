# Spellbook

AI-powered Magic: The Gathering deck builder for competitive Standard play.

## Overview

Spellbook is a full-stack application that uses AI to generate tournament-ready Magic: The Gathering decks. It combines real-time metagame data with Claude AI to create optimized decks tailored to the current competitive landscape.

## Features

### Deck Building
- **AI-Powered Generation**: Describe your desired deck strategy and receive a complete, validated deck
- **Anti-Hallucination System**: AI only selects from real, Standard-legal cards in the database
- **Deck Validation**: Automatic checking of deck size, copy limits, and format legality
- **Import/Export**: Support for Arena, MTGO, and plain text formats
- **Iteration**: Refine your deck through natural language conversation

### Metagame Analysis
- **Live Meta Data**: Weekly scraping of tournament results from mtgtop8
- **Archetype Breakdown**: View meta percentages, average finishes, and key cards
- **Card Co-occurrence**: Understand which cards are commonly played together
- **Matchup Analysis**: AI-generated matchup ratings against top archetypes

### User Features
- **Account Management**: Registration, email verification, password reset
- **Profile Customization**: Avatar upload, display name, email change
- **Deck Library**: Save, organize, and share your decks
- **Conversation History**: Persist AI conversations across sessions
- **Preferences**: Theme (light/dark), language, default format

### Operations
- **Scheduled Jobs**: Daily Scryfall sync (2 AM UTC), weekly mtgtop8 scrape (Sunday 6 AM UTC)
- **Retry Logic**: Exponential backoff (1m, 2m, 4m) with 3 attempts
- **Alert Notifications**: Email alerts on job failures
- **Admin Dashboard**: Job health monitoring, execution history, manual triggers

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL with pgvector for semantic search
- **ORM**: SQLAlchemy (async)
- **Migrations**: Alembic
- **Job Scheduler**: APScheduler
- **AI**: Anthropic Claude API

### Frontend
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **HTTP Client**: Axios with interceptors
- **Routing**: React Router v6

### Infrastructure
- **Containerization**: Docker & docker-compose
- **Email**: SendGrid (configurable)

## Project Structure

```
3-Spellbook/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/      # API endpoints
│   │   │   └── deps/        # Dependencies (auth, db)
│   │   ├── core/            # Config, security
│   │   ├── db/              # Database session
│   │   ├── jobs/            # Scheduled jobs
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   └── utils/           # Helpers
│   ├── alembic/             # Database migrations
│   ├── tests/               # Backend tests
│   └── uploads/             # User uploads (avatars)
├── frontend/
│   └── src/
│       ├── components/      # React components
│       ├── hooks/           # Custom hooks
│       ├── pages/           # Route pages
│       ├── services/        # API client
│       ├── store/           # Zustand stores
│       └── types/           # TypeScript types
└── docker-compose.yml
```

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+ with pgvector extension
- Docker (optional)

### Environment Variables

Create `.env` files in both `backend/` and `frontend/` directories:

**Backend (.env)**
```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/spellbook
SECRET_KEY=your-secret-key
ANTHROPIC_API_KEY=your-anthropic-key
SENDGRID_API_KEY=your-sendgrid-key
APP_URL=http://localhost:8000
ENABLE_SCHEDULER=true
ALERT_EMAIL=admin@example.com
```

**Frontend (.env)**
```env
VITE_API_URL=http://localhost:8000
```

### Installation

**Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker-compose up --build
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login
- `POST /api/auth/refresh` - Refresh token
- `POST /api/auth/verify/{token}` - Verify email
- `POST /api/auth/password-reset/request` - Request password reset
- `POST /api/auth/password-reset/confirm` - Confirm password reset

### Users
- `GET /api/users/me` - Get current user
- `PATCH /api/users/me` - Update profile
- `POST /api/users/me/avatar` - Upload avatar
- `POST /api/users/me/change-password` - Change password
- `POST /api/users/me/email-change` - Request email change
- `GET /api/users/me/preferences` - Get preferences
- `PATCH /api/users/me/preferences` - Update preferences

### Cards
- `GET /api/cards/search` - Search cards with filters
- `GET /api/cards/{id}` - Get card by ID
- `POST /api/cards/semantic-search` - Natural language search
- `POST /api/cards/ai/candidate-cards` - Get candidates for AI selection

### Decks
- `GET /api/decks` - List user's decks
- `POST /api/decks` - Create deck
- `GET /api/decks/{id}` - Get deck
- `PATCH /api/decks/{id}` - Update deck
- `DELETE /api/decks/{id}` - Delete deck
- `GET /api/decks/public/{token}` - Get shared deck
- `POST /api/decks/import` - Import decklist
- `GET /api/decks/{id}/export` - Export deck
- `POST /api/decks/generate` - AI generate deck
- `POST /api/decks/iterate` - AI iterate on deck

### Conversations
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation
- `POST /api/conversations/chat` - Send message
- `POST /api/conversations/explain-card` - Explain card
- `DELETE /api/conversations/{id}` - Delete conversation

### Meta
- `GET /api/meta` - Get meta dashboard
- `GET /api/meta/archetypes/{name}` - Get archetype details
- `GET /api/meta/cooccurrence/{card}` - Get card co-occurrence

### Admin
- `POST /api/admin/jobs/{name}/run` - Trigger job manually
- `GET /api/admin/jobs/history` - Get job history
- `GET /api/admin/dashboard/jobs` - Get job metrics

### Health
- `GET /health` - Basic health check
- `GET /health/jobs` - Job health status

## License

MIT
