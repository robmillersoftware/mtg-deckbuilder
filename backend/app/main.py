from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.session import init_db
from app.api.routes import (
    auth,
    users,
    cards,
    decks,
    conversations,
    meta,
    health,
    admin,
    simulation,
    guided_build,
)
from app.jobs.scheduler import configure_scheduler, start_scheduler, shutdown_scheduler

# Ensure upload directories exist
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(os.path.join(UPLOADS_DIR, "avatars"), exist_ok=True)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()

    # Cleanup stale simulation runs (from server restarts/crashes)
    await cleanup_stale_simulations()

    # Start scheduler if enabled
    if settings.ENABLE_SCHEDULER:
        configure_scheduler()
        start_scheduler()
        logger.info("Job scheduler started")

    yield

    # Shutdown
    if settings.ENABLE_SCHEDULER:
        shutdown_scheduler()
        logger.info("Job scheduler stopped")


async def cleanup_stale_simulations():
    """Mark any orphaned 'running' simulations as failed on startup."""
    from app.db.session import async_session_factory
    from app.services.game_simulator import GameSimulator

    try:
        async with async_session_factory() as db:
            simulator = GameSimulator(db)
            count = await simulator.cleanup_stale_runs(timeout_minutes=10)
            if count > 0:
                logger.info(f"Cleaned up {count} stale simulation run(s)")
    except Exception as e:
        logger.error(f"Failed to cleanup stale simulations: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered Magic: The Gathering deck builder",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", settings.APP_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(cards.router, prefix="/api/cards", tags=["Cards"])
app.include_router(decks.router, prefix="/api/decks", tags=["Decks"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["Conversations"])
app.include_router(meta.router, prefix="/api/meta", tags=["Meta"])
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(guided_build.router, prefix="/api/guided-build", tags=["Guided Build"])

# Mount static files directory for uploads
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "description": "AI-powered Magic: The Gathering deck builder",
    }
