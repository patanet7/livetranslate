import asyncio
import contextlib
import json
import os
from datetime import UTC, datetime

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..database.seamless_storage import (
    add_event,
    close_session,
    ensure_schema,
    get_events,
    get_session,
    get_transcripts,
    list_sessions,
    open_session,
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
    forward_task: asyncio.Task | None = None

    try:
        # Ensure schema and open session record
        await ensure_schema()
        await open_session(
            session_id=session_id,
            created_at_iso=datetime.now(UTC).isoformat(),
            source_lang="cmn",
            target_lang="eng",
            client_ip=str(websocket.client.host) if websocket.client else None,
            user_agent=websocket.headers.get("user-agent"),
        )
        # Connect to Seamless service
        seamless_ws = await websockets.connect(upstream, max_size=2**24)

        # Start forwarder from upstream to client
        forward_task = asyncio.create_task(_forward_from_seamless_to_client(seamless_ws, websocket))

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
                await add_event(session_id, etype, msg, int(datetime.now(UTC).timestamp() * 1000))

    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Seamless proxy error: {e}",
                    }
                )
            )
    finally:
        if forward_task and not forward_task.done():
            forward_task.cancel()
        if seamless_ws is not None:
            with contextlib.suppress(Exception):
                await seamless_ws.close()
        # Close session
        with contextlib.suppress(Exception):
            await close_session(session_id, datetime.now(UTC).isoformat())
        with contextlib.suppress(Exception):
            await websocket.close()


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
