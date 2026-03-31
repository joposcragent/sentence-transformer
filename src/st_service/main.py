from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
import numpy as np

from st_service.config import get_settings, resolve_torch_device
from st_service.cosine import cosine_similarity_01
from st_service.embedding import encode_text, load_model, max_context_tokens, token_length
from st_service.schemas import CosineSimilarityResponse, TextCorpus, VectorsPair

API_PREFIX = "/sentence-transformer"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    device = resolve_torch_device(settings.st_device)
    cache = str(settings.st_model_cache_dir) if settings.st_model_cache_dir else None
    app.state.model = load_model(
        settings.st_model_name,
        device=device,
        cache_folder=cache,
    )
    yield


app = FastAPI(
    title="Job posting scraping agent - sentence transformer",
    version="1.0.0",
    lifespan=lifespan,
)

router = APIRouter(prefix=API_PREFIX)


@router.get("/health")
def health(request: Request):
    try:
        _ = request.app.state.model
        return {"status": "READY"}
    except Exception as exc:
        return PlainTextResponse(str(exc), status_code=503)


@router.post("/text/vectorize")
def vectorize(request: Request, body: TextCorpus):
    model = request.app.state.model
    try:
        n = token_length(model, body.text)
        limit = max_context_tokens(model)
        if n > limit:
            raise HTTPException(status_code=413)
        vec = encode_text(model, body.text)
        return vec.tolist()
    except HTTPException:
        raise
    except Exception as exc:
        return PlainTextResponse(str(exc), status_code=500)


@router.post("/vectors/cosine-similarity")
def cosine_similarity_endpoint(body: VectorsPair):
    try:
        left = np.asarray(body.left, dtype=np.float64)
        right = np.asarray(body.right, dtype=np.float64)
        sim = cosine_similarity_01(left, right)
        return CosineSimilarityResponse(similarity=sim)
    except Exception as exc:
        return PlainTextResponse(str(exc), status_code=500)


app.include_router(router)
