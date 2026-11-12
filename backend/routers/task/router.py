from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask
import os
from typing import List

from backend.db.db_instance import get_db_session
from backend.db.task.dao import (
    get_task_status_from_db,
    get_all_tasks_status_from_db,
    delete_task_from_db,
)
from backend.db.task.models import (
    TasksResult,
    Task,
    TaskStatusResponse,
    TaskStatus,
    TaskType
)
from backend.common.models import (
    Response,
)
from backend.common.compresser import (
    create_download_archive,
    find_file_by_hash,
    get_file_hash,
    resolve_cache_artifact,
)
from modules.utils.logger import get_logger
from modules.utils.paths import BACKEND_CACHE_DIR

task_router = APIRouter(prefix="/task", tags=["Tasks"])

logger = get_logger()

ARTIFACT_HASH_KEYS = {"instrumental": "instrumental_hash", "vocal": "vocal_hash"}
LEGACY_ARTIFACT_DIRS = {"instrumental": "instrumental", "vocal": "vocals"}
UNFINISHED_STATUSES = (
    TaskStatus.QUEUED,
    TaskStatus.PENDING,
    TaskStatus.IN_PROGRESS,
    TaskStatus.RETRYING,
    TaskStatus.PAUSED,
)


def _remove_archive(path: str) -> None:
    """Delete this request's archive; it belongs to this response only."""
    try:
        os.remove(path)
    except OSError as e:
        logger.warning(f"Could not remove the download archive {path}: {e}")


def resolve_task_artifacts(task: Task) -> List[str]:
    """Return the task's artifact paths, verified against the recorded hashes.

    New tasks store cache-relative references, so the cache is not hashed on every
    download; records from older releases fall back to the hash lookup. A missing,
    expired, replaced or unsafe artifact is reported with a predictable status.
    """
    references = task.artifacts or {}
    result = task.result or {}
    paths: List[str] = []

    for kind, hash_key in ARTIFACT_HASH_KEYS.items():
        expected_hash = result.get(hash_key)
        if not expected_hash:
            raise HTTPException(
                status_code=422,
                detail=f"The task result has no '{hash_key}', so its artifacts cannot be downloaded.",
            )

        relative = references.get(kind)
        if relative:
            path = resolve_cache_artifact(BACKEND_CACHE_DIR, relative)
            if path is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"The task references a '{kind}' artifact outside the cache directory.",
                )
        else:
            directory = os.path.join(BACKEND_CACHE_DIR, "UVR", LEGACY_ARTIFACT_DIRS[kind])
            try:
                path = find_file_by_hash(directory, expected_hash)
            except FileNotFoundError:
                path = None

        if path is None or not os.path.isfile(path):
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"The {kind} artifact is no longer available (the cache TTL expired it).",
            )
        try:
            actual_hash = get_file_hash(path)
        except OSError:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"The {kind} artifact exists but cannot be read.",
            )
        if actual_hash != expected_hash:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"The {kind} artifact at its recorded path no longer matches its hash.",
            )
        paths.append(path)

    return paths


@task_router.get(
    "/{identifier}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Task by Identifier",
    description="Retrieve the specific task by its identifier.",
)
async def get_task(
    identifier: str,
    session: Session = Depends(get_db_session),
) -> TaskStatusResponse:
    """
    Retrieve the specific task by its identifier.
    """
    task = get_task_status_from_db(identifier=identifier, session=session)

    if task is not None:
        return task.to_response()
    else:
        raise HTTPException(status_code=404, detail="Identifier not found")


@task_router.get(
    "/file/{identifier}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve FileResponse Task by Identifier",
    description="Retrieve the file response task by its identifier. You can use this endpoint if you need to download"
                " The file as a response. Only BGM separation tasks have downloadable artifacts: 404 for an unknown"
                " identifier or another task type, 409 while the task is queued/running/failed/cancelled, and 410 once"
                " the cache TTL has expired its files. Cache expiry is separate from task success - a task stays"
                " COMPLETED after its files are cleaned up, and its status endpoint keeps reporting the result.",
)
async def get_file_task(
    identifier: str,
    session: Session = Depends(get_db_session),
) -> FileResponse:
    """
    Retrieve the downloadable file response of a specific task by its identifier.
    Compressed by ZIP basically.
    """
    task = get_task_status_from_db(identifier=identifier, session=session)

    if task is None:
        raise HTTPException(status_code=404, detail="Identifier not found")

    if task.task_type != TaskType.BGM_SEPARATION:
        raise HTTPException(status_code=404, detail=f"File download is only supported for bgm separation."
                                                    f" The given type is {task.task_type}")

    if task.status in UNFINISHED_STATUSES:
        raise HTTPException(status_code=409, detail=f"Task is '{task.status}'; its artifacts are not available yet.")
    if task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
        raise HTTPException(status_code=409, detail=f"Task is '{task.status}'; it produced no downloadable artifacts.")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Task is '{task.status}'; no downloadable artifacts are defined.")

    artifact_paths = resolve_task_artifacts(task)

    archive_path = create_download_archive(artifact_paths, BACKEND_CACHE_DIR, f"{identifier}_bgm_separation")

    return FileResponse(
        path=archive_path,
        status_code=200,
        filename=f"{identifier}_bgm_separation.zip",
        media_type="application/zip",
        background=BackgroundTask(_remove_archive, archive_path),
    )


# Delete method, commented by default because this endpoint is likely to require special permissions
# @task_router.delete(
#     "/{identifier}",
#     response_model=Response,
#     status_code=status.HTTP_200_OK,
#     summary="Delete Task by Identifier",
#     description="Delete a task from the system using its identifier.",
# )
async def delete_task(
    identifier: str,
    session: Session = Depends(get_db_session),
) -> Response:
    """
    Delete a task by its identifier.
    """
    if delete_task_from_db(identifier, session):
        return Response(identifier=identifier, message="Task deleted")
    else:
        raise HTTPException(status_code=404, detail="Task not found")


# Get All method, commented by default because this endpoint is likely to require special permissions
# @task_router.get(
#     "/all",
#     response_model=TasksResult,
#     status_code=status.HTTP_200_OK,
#     summary="Retrieve All Task Statuses",
#     description="Retrieve the statuses of all tasks available in the system.",
# )
async def get_all_tasks_status(
    session: Session = Depends(get_db_session),
) -> TasksResult:
    """
    Retrieve all tasks.
    """
    return get_all_tasks_status_from_db(session=session)
