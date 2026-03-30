"""FastAPI 应用入口 — Financial RAG REST API。"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import documents, query

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial RAG API",
    description="金融领域 RAG 知识库问答系统 REST API",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# 中间件
# ---------------------------------------------------------------------------

# CORS：允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """请求日志中间件：记录每个 API 调用的方法和延迟。"""
    t0 = time.perf_counter()
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("%s %s → %d (%.1fms)", method, path, response.status_code, latency_ms)
        return response
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("%s %s → 500 (%.1fms): %s", method, path, latency_ms, e)
        raise


# ---------------------------------------------------------------------------
# 全局异常处理
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """统一错误响应格式。"""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"内部服务器错误: {exc}"},
    )


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

app.include_router(query.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# 启动/关闭事件
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    logger.info("Financial RAG API 启动中...")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Financial RAG API 关闭")
