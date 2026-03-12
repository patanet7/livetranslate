"""REST API for unified AI connections management."""

import json

from fastapi import APIRouter, Depends, HTTPException
from livetranslate_common.logging import get_logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db_session
from database.models import SystemConfig
from models.connections import (
    AIConnectionCreate,
    AIConnectionResponse,
    AIConnectionUpdate,
    AggregateModelsResponse,
    FeaturePreference,
    VerifyResult,
)
from services.connections import ConnectionService

logger = get_logger()
router = APIRouter(prefix="/api/connections", tags=["connections"])

_VALID_FEATURES = {"chat", "translation", "intelligence"}


def get_connection_service(db: AsyncSession = Depends(get_db_session)) -> ConnectionService:
    return ConnectionService(db)


# ── Fixed-path routes MUST come before {connection_id} to avoid path capture ──

@router.post("/aggregate-models", response_model=AggregateModelsResponse)
async def aggregate_models(
    svc: ConnectionService = Depends(get_connection_service),
):
    return await svc.aggregate_models()


@router.get("/preferences/all")
async def get_all_preferences(
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, FeaturePreference]:
    result = {}
    for feature in _VALID_FEATURES:
        key = f"{feature}_model_preference"
        row = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
        config = row.scalar_one_or_none()
        if config and config.value:
            data = json.loads(config.value) if isinstance(config.value, str) else config.value
            result[feature] = FeaturePreference(**data)
        else:
            result[feature] = FeaturePreference()
    return result


@router.put("/preferences/{feature}", response_model=FeaturePreference)
async def update_feature_preference(
    feature: str,
    pref: FeaturePreference,
    db: AsyncSession = Depends(get_db_session),
):
    if feature not in _VALID_FEATURES:
        raise HTTPException(status_code=400, detail=f"feature must be one of {_VALID_FEATURES}")
    key = f"{feature}_model_preference"
    val = json.dumps(pref.model_dump())
    row = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
    config = row.scalar_one_or_none()
    if config:
        config.value = val
    else:
        db.add(SystemConfig(key=key, value=val))
    await db.commit()
    return pref


# ── Collection routes ──

@router.get("", response_model=list[AIConnectionResponse])
async def list_connections(
    enabled_only: bool = False,
    svc: ConnectionService = Depends(get_connection_service),
):
    connections = await svc.list_connections(enabled_only=enabled_only)
    return [svc.to_response(c) for c in connections]


@router.post("", response_model=AIConnectionResponse, status_code=201)
async def create_connection(
    data: AIConnectionCreate,
    svc: ConnectionService = Depends(get_connection_service),
):
    conn = await svc.create_connection(data)
    return svc.to_response(conn)


# ── Parameterized routes ──

@router.get("/{connection_id}", response_model=AIConnectionResponse)
async def get_connection(
    connection_id: str,
    svc: ConnectionService = Depends(get_connection_service),
):
    conn = await svc.get_connection(connection_id)
    return svc.to_response(conn)


@router.put("/{connection_id}", response_model=AIConnectionResponse)
async def update_connection(
    connection_id: str,
    data: AIConnectionUpdate,
    svc: ConnectionService = Depends(get_connection_service),
):
    conn = await svc.update_connection(connection_id, data)
    return svc.to_response(conn)


@router.delete("/{connection_id}", status_code=204)
async def delete_connection(
    connection_id: str,
    svc: ConnectionService = Depends(get_connection_service),
):
    await svc.delete_connection(connection_id)


@router.post("/{connection_id}/verify", response_model=VerifyResult)
async def verify_connection(
    connection_id: str,
    svc: ConnectionService = Depends(get_connection_service),
):
    return await svc.verify_connection(connection_id)
