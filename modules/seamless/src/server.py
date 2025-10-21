import asyncio
import base64
import json
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .streaming_st import StreamingTranslator, StreamingConfig

app = FastAPI(title="Seamless Demo Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionState:
    def __init__(self, session_id: str, src_lang: str = "cmn", tgt_lang: str = "eng") -> None:
        self.session_id = session_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.translator = StreamingTranslator(StreamingConfig(source_lang=src_lang, target_lang=tgt_lang))
        self._partial_task: asyncio.Task | None = None

    async def send_partial_if_any(self, websocket: WebSocket) -> None:
        text = self.translator.infer_partial()
        if text:
            await websocket.send_text(
                json.dumps({
                    "type": "translation_partial",
                    "lang": "en",
                    "text": text,
                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                })
            )


async def _periodic_partial(websocket: WebSocket, state: SessionState) -> None:
    try:
        while True:
            await asyncio.sleep(0.8)
            await state.send_partial_if_any(websocket)
    except asyncio.CancelledError:
        return


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, session_id: str | None = None) -> None:
    await websocket.accept()

    sid = session_id or f"session-{int(datetime.utcnow().timestamp())}"
    state = SessionState(session_id=sid)

    # Start periodic mock partials
    state._partial_task = asyncio.create_task(_periodic_partial(websocket, state))

    try:
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": sid,
            "message": "Seamless demo WS connected",
            "timestamp": datetime.utcnow().isoformat(),
        }))

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "config":
                state.src_lang = msg.get("source", state.src_lang) or state.src_lang
                state.tgt_lang = msg.get("target", state.tgt_lang) or state.tgt_lang
                # Recreate translator if languages changed
                state.translator = StreamingTranslator(StreamingConfig(source_lang=state.src_lang, target_lang=state.tgt_lang))
                await websocket.send_text(json.dumps({
                    "type": "config_ack",
                    "source": state.src_lang,
                    "target": state.tgt_lang,
                    "timestamp": datetime.utcnow().isoformat(),
                }))

            elif mtype == "audio_chunk":
                b64 = msg.get("data", "")
                try:
                    decoded = base64.b64decode(b64)
                    state.translator.add_audio_pcm16(decoded)
                except Exception:
                    # Ignore malformed input in demo
                    pass

            elif mtype == "end":
                final_text = state.translator.infer_final()
                await websocket.send_text(json.dumps({
                    "type": "translation_final",
                    "lang": "en",
                    "text": final_text,
                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                }))
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e),
        }))
    finally:
        if state._partial_task and not state._partial_task.done():
            state._partial_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


