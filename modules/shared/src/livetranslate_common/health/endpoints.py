"""Health check endpoint factory."""

from collections.abc import Callable

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


def create_health_router(
    service_name: str,
    version: str,
    checks: dict[str, Callable[[], bool]],
) -> APIRouter:
    """Return a router with a ``/health`` endpoint that runs all checks.

    Args:
        service_name: Human-readable service identifier included in the response.
        version: Semantic version string included in the response.
        checks: Mapping of check name to a callable that returns ``True`` when
            the dependency is healthy. If any callable returns ``False`` or raises
            an exception the overall status is ``unhealthy`` and the endpoint
            returns HTTP 503.

    Returns:
        A FastAPI ``APIRouter`` with a single ``GET /health`` route.
    """
    router = APIRouter()

    @router.get("/health")
    async def health() -> JSONResponse:
        results: dict[str, str] = {}
        all_ok = True
        for name, check_fn in checks.items():
            try:
                ok = check_fn()
            except Exception:
                logger.warning("health_check_failed", check=name, exc_info=True)
                ok = False
            results[name] = "ok" if ok else "failing"
            if not ok:
                all_ok = False

        status = "healthy" if all_ok else "unhealthy"
        status_code = 200 if all_ok else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": status,
                "service": service_name,
                "version": version,
                "checks": results,
            },
        )

    return router
