from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from st_service import __version__
from st_service.config import get_settings, resolve_torch_device
from st_service.cosine import cosine_similarity_01
from st_service.embedding import encode_text, load_model, max_context_tokens, token_length
from st_service.log_preview import preview_text, preview_vector
from st_service.schemas import CosineSimilarityResponse, TextCorpus, VectorsPair

logger = logging.getLogger(__name__)

API_PREFIX = "/sentence-transformer"


class _SuppressHealthAccessLogFilter(logging.Filter):
    """Drop uvicorn access lines for GET <API_PREFIX>/health (path may include query)."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        method, raw_path = args[1], args[2]
        if method != "GET":
            return True
        path_only = str(raw_path).split("?", 1)[0]
        if path_only == f"{API_PREFIX}/health":
            return False
        return True


def _install_uvicorn_access_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if any(isinstance(f, _SuppressHealthAccessLogFilter) for f in access_logger.filters):
        return
    access_logger.addFilter(_SuppressHealthAccessLogFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    device = resolve_torch_device(settings.st_device)
    cache = str(settings.st_model_cache_dir) if settings.st_model_cache_dir else None
    logger.info(
        "Loading sentence-transformer model name=%s device=%s cache_folder=%s",
        settings.st_model_name,
        device,
        cache or "(default)",
    )
    try:
        app.state.model = load_model(
            settings.st_model_name,
            device=device,
            cache_folder=cache,
        )
    except Exception:
        logger.exception("Failed to load sentence-transformer model")
        raise
    logger.info(
        "Model ready name=%s device=%s cache_folder=%s",
        settings.st_model_name,
        device,
        cache or "(default)",
    )
    yield


app = FastAPI(
    title="Job posting scraping agent - sentence transformer",
    version=__version__,
    lifespan=lifespan,
)

router = APIRouter(prefix=API_PREFIX)


@router.get("/health")
def health(request: Request):
    try:
        _ = request.app.state.model
        return {"status": "READY"}
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return PlainTextResponse(str(exc), status_code=503)


@router.post("/text/vectorize")
def vectorize(request: Request, body: TextCorpus):
    model = request.app.state.model
    try:
        n = token_length(model, body.text)
        limit = max_context_tokens(model)
        logger.info(
            "Vectorize start text_len=%s token_count=%s token_limit=%s text_preview=%r",
            len(body.text),
            n,
            limit,
            preview_text(body.text),
        )
        if n > limit:
            logger.warning(
                "Vectorize rejected 413 token_count=%s token_limit=%s text_preview=%r",
                n,
                limit,
                preview_text(body.text),
            )
            raise HTTPException(status_code=413)
        vec = encode_text(model, body.text)
        logger.info(
            "Vectorize success dim=%s vector_preview=%r",
            vec.size,
            preview_vector(vec),
        )
        return vec.tolist()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Vectorize failed")
        return PlainTextResponse(str(exc), status_code=500)


@router.post("/vectors/cosine-similarity")
def cosine_similarity_endpoint(body: VectorsPair):
    try:
        left = np.asarray(body.left, dtype=np.float64)
        right = np.asarray(body.right, dtype=np.float64)
        logger.info(
            "Cosine similarity start left_len=%s right_len=%s left_preview=%r right_preview=%r",
            left.size,
            right.size,
            preview_vector(left),
            preview_vector(right),
        )
        sim = cosine_similarity_01(left, right)
        logger.info("Cosine similarity success similarity=%s", sim)
        return CosineSimilarityResponse(similarity=sim)
    except Exception as exc:
        logger.exception("Cosine similarity failed")
        return PlainTextResponse(str(exc), status_code=500)


app.include_router(router)

_install_uvicorn_access_log_filter()
