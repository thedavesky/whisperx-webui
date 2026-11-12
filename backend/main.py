from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
)
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import threading

from backend.db.db_instance import init_db
from backend.db.task.dao import reconcile_interrupted_tasks
from backend.routers.transcription.router import transcription_router, get_pipeline
from backend.routers.bgm_separation.router import get_bgm_separation_inferencer, bgm_separation_router
from backend.routers.task.router import task_router
from backend.common.config_loader import read_env, load_server_config
from backend.common.cache_manager import cleanup_old_files
from modules.utils.logger import get_logger
from modules.utils.paths import SERVER_CONFIG_PATH, BACKEND_CACHE_DIR


logger = get_logger()


def clean_cache_thread(ttl: int, frequency: int, stop: threading.Event) -> threading.Thread:
    def clean_cache():
        while not stop.wait(frequency):
            cleanup_old_files(cache_dir=BACKEND_CACHE_DIR, ttl=ttl)

    return threading.Thread(
        target=clean_cache,
        daemon=True
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Basic setup initialization
    server_config = load_server_config()
    read_env("DB_URL")  # Place .env file into /configs/.env
    init_db()

    # Jobs run in this process, so tasks from an earlier process can never resume.
    reconcile_interrupted_tasks()

    # Inferencer initialization
    transcription_pipeline = get_pipeline()
    bgm_separation_inferencer = get_bgm_separation_inferencer()

    # Thread initialization
    stop_cache = threading.Event()
    cache_thread = clean_cache_thread(server_config["cache"]["ttl"], server_config["cache"]["frequency"], stop_cache)
    cache_thread.start()

    yield

    # Release VRAM, the worker processes and the sweep thread when the server shuts down.
    stop_cache.set()
    cache_thread.join(timeout=5)
    if cache_thread.is_alive():
        logger.warning("Cache cleanup thread is still running after 5s; it is a daemon thread.")

    for inferencer in (transcription_pipeline, bgm_separation_inferencer):
        try:
            inferencer.offload()
        except Exception as e:
            logger.error(f"Failed to offload {type(inferencer).__name__} on shutdown: {e}")

    # Drop the cached instances so a restarted app in the same process rebuilds them.
    get_pipeline.cache_clear()
    get_bgm_separation_inferencer.cache_clear()


app = FastAPI(
    title="Whisper-WebUI-Backend",
    description=f"""
    REST API for Whisper-WebUI. Swagger UI is available via /docs or root URL with redirection. Redoc is available via /redoc. 
    """,
    version="0.0.1",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "BGM Separation",
            "description": "Cached files for /bgm-separation are generated in the `backend/cache` directory,"
                           " you can set TLL for these files in `backend/configs/config.yaml`."
        }
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],  # Disable DELETE
    allow_headers=["*"],
)
app.include_router(transcription_router)
app.include_router(bgm_separation_router)
app.include_router(task_router)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    """
    Redirect to the documentation. Defaults to Swagger UI.
    You can also check the /redoc with redoc style: https://github.com/Redocly/redoc
    """
    return "/docs"
