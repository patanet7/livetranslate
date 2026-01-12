import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Optional

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..database.seamless_storage import (
    ensure_schema,
    open_session,
    close_session,
    add_event,
    list_sessions,
    get_session,
    get_events,
    get_transcripts,
)


router = APIRouter()


async def _forward_from_seamless_to_client(seamless_ws, client_ws: WebSocket):
    try:
        async for message in seamless_ws:
            await client_ws.send_text(message)
    except Exception:
        # Upstream ended or client closed
        return


@router.websocket("/realtime/{session_id}")
async def seamless_realtime(websocket: WebSocket, session_id: str):
    await websocket.accept()

    upstream_url = os.getenv("SEAMLESS_WS_URL", "ws://localhost:5007/ws/stream")
    # Append session_id param
    sep = "&" if "?" in upstream_url else "?"
    upstream = f"{upstream_url}{sep}session_id={session_id}"

    seamless_ws = None
    forward_task: Optional[asyncio.Task] = None

    try:
        # Ensure schema and open session record
        await ensure_schema()
        await open_session(
            session_id=session_id,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            source_lang="cmn",
            target_lang="eng",
            client_ip=str(websocket.client.host) if websocket.client else None,
            user_agent=websocket.headers.get("user-agent"),
        )
        # Connect to Seamless service
        seamless_ws = await websockets.connect(upstream, max_size=2**24)

        # Start forwarder from upstream to client
        forward_task = asyncio.create_task(
            _forward_from_seamless_to_client(seamless_ws, websocket)
        )

        # Inform upstream of connection
        await seamless_ws.send(
            json.dumps(
                {
                    "type": "config",
                    "source": "cmn",
                    "target": "eng",
                    "emitPartials": True,
                }
            )
        )

        # Main loop: receive client messages and forward upstream
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            msg = json.loads(raw)
            # Forward all messages as-is
            await seamless_ws.send(json.dumps(msg))
            # Persist event (throttle client audio chunks if needed)
            etype = msg.get("type", "unknown")
            if etype != "audio_chunk":
                await add_event(
                    session_id, etype, msg, int(datetime.now(timezone.utc).timestamp() * 1000)
                )

    except Exception as e:
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Seamless proxy error: {e}",
                    }
                )
            )
        except Exception:
            pass
    finally:
        if forward_task and not forward_task.done():
            forward_task.cancel()
        if seamless_ws is not None:
            try:
                await seamless_ws.close()
            except Exception:
                pass
        # Close session
        try:
            await close_session(session_id, datetime.now(timezone.utc).isoformat())
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


# Retrieval APIs
@router.get("/sessions")
async def http_list_sessions(limit: int = 50, offset: int = 0):
    await ensure_schema()
    return await list_sessions(limit=limit, offset=offset)


@router.get("/sessions/{session_id}")
async def http_get_session(session_id: str):
    await ensure_schema()
    return await get_session(session_id)


@router.get("/sessions/{session_id}/events")
async def http_get_events(
    session_id: str,
    from_ms: int | None = None,
    to_ms: int | None = None,
    types: str | None = None,
    limit: int = 500,
    offset: int = 0,
):
    await ensure_schema()
    return await get_events(session_id, from_ms, to_ms, types, limit, offset)


@router.get("/sessions/{session_id}/transcripts")
async def http_get_transcripts(session_id: str):
    await ensure_schema()
    return await get_transcripts(session_id)
