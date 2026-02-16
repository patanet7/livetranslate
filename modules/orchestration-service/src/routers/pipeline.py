"""
Pipeline Processing API Router

Real-time audio pipeline processing endpoints for the Pipeline Studio
"""

import contextlib
import json
import time
import uuid
from typing import Any

from dependencies import (
    get_audio_coordinator,
    get_config_manager,
)
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from livetranslate_common.logging import get_logger
from pydantic import BaseModel, ValidationError

logger = get_logger()

router = APIRouter()


# Pydantic Models
class PipelineStageConfig(BaseModel):
    enabled: bool = True
    gain_in: float = 0.0  # -20 to +20 dB
    gain_out: float = 0.0  # -20 to +20 dB
    parameters: dict[str, Any] = {}


class PipelineConnection(BaseModel):
    id: str
    source_stage_id: str
    target_stage_id: str


class PipelineConfig(BaseModel):
    pipeline_id: str
    name: str
    stages: dict[str, PipelineStageConfig]
    connections: list[PipelineConnection]


class PipelineProcessingRequest(BaseModel):
    pipeline_config: PipelineConfig
    processing_mode: str = "batch"  # "batch", "realtime", "preview"
    output_format: str = "wav"  # "wav", "mp3", "base64"
    metadata: dict[str, Any] | None = None


class PipelineProcessingResponse(BaseModel):
    success: bool
    pipeline_id: str
    processed_audio: str | None = None  # Base64 encoded
    metrics: dict[str, Any]
    stage_outputs: dict[str, str] | None = None  # Base64 encoded audio for each stage
    errors: list[str] | None = None
    warnings: list[str] | None = None


class RealtimeSessionConfig(BaseModel):
    chunk_size: int = 1024
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 4096
    latency_target: int = 100  # milliseconds


class RealtimeSession(BaseModel):
    session_id: str
    pipeline_id: str
    status: str = "initializing"  # "initializing", "running", "paused", "stopped", "error"
    metrics: dict[str, Any] = {}


# Global state for WebSocket connections
active_sessions: dict[str, dict[str, Any]] = {}


@router.post("/process", response_model=PipelineProcessingResponse)
async def process_pipeline(
    pipeline_config: str = Form(...),
    audio_file: UploadFile | None = File(None),
    audio_data: str | None = Form(None),
    processing_mode: str = Form("batch"),
    output_format: str = Form("wav"),
    metadata: str | None = Form(None),
    audio_coordinator=Depends(get_audio_coordinator),
    config_manager=Depends(get_config_manager),
):
    """
    Process audio through a complete pipeline configuration
    """
    try:
        # Parse pipeline configuration
        try:
            pipeline_config_dict = json.loads(pipeline_config)
            pipeline = PipelineConfig(**pipeline_config_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid pipeline configuration: {e!s}",
            ) from e

        # Get audio data
        audio_content = None
        if audio_file:
            audio_content = await audio_file.read()
        elif audio_data:
            import base64

            audio_content = base64.b64decode(audio_data)

        if not audio_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No audio data provided"
            )

        # Parse metadata
        if metadata:
            try:
                json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON, using empty dict")

        logger.info(
            f"Processing pipeline {pipeline.pipeline_id} with {len(pipeline.stages)} stages"
        )

        # Convert pipeline config to orchestration format
        orchestration_config = await convert_pipeline_to_orchestration_format(pipeline)

        # Process through audio coordinator
        start_time = time.time()

        try:
            # Apply pipeline configuration to audio coordinator
            await audio_coordinator.apply_pipeline_config(orchestration_config)

            # Process audio
            processed_result = await audio_coordinator.process_audio_pipeline(
                audio_content,
                pipeline_config=orchestration_config,
                output_format=output_format,
            )

            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Calculate metrics
            metrics = {
                "total_latency": processing_time,
                "stage_latencies": processed_result.get("stage_latencies", {}),
                "quality_metrics": {
                    "snr": processed_result.get("snr", 0),
                    "thd": processed_result.get("thd", 0),
                    "lufs": processed_result.get("lufs", -23.0),
                    "rms": processed_result.get("rms", -20.0),
                },
                "cpu_usage": processed_result.get("cpu_usage", 0),
            }

            # Encode processed audio to base64
            processed_audio_b64 = None
            if processed_result.get("processed_audio"):
                import base64

                processed_audio_b64 = base64.b64encode(processed_result["processed_audio"]).decode(
                    "utf-8"
                )

            # Get stage outputs if available
            stage_outputs = {}
            if processed_result.get("stage_outputs"):
                for stage_name, stage_audio in processed_result["stage_outputs"].items():
                    stage_outputs[stage_name] = base64.b64encode(stage_audio).decode("utf-8")

            logger.info(f"Pipeline processing completed in {processing_time:.1f}ms")

            return PipelineProcessingResponse(
                success=True,
                pipeline_id=pipeline.pipeline_id,
                processed_audio=processed_audio_b64,
                metrics=metrics,
                stage_outputs=stage_outputs if stage_outputs else None,
                errors=processed_result.get("errors"),
                warnings=processed_result.get("warnings"),
            )

        except Exception as processing_error:
            logger.error(f"Pipeline processing failed: {processing_error!s}")
            return PipelineProcessingResponse(
                success=False,
                pipeline_id=pipeline.pipeline_id,
                metrics={"total_latency": (time.time() - start_time) * 1000},
                errors=[f"Processing failed: {processing_error!s}"],
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in pipeline processing: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e!s}",
        ) from e


@router.post("/realtime/start", response_model=RealtimeSession)
async def start_realtime_session(
    request: dict[str, Any],
    audio_coordinator=Depends(get_audio_coordinator),
):
    """
    Start a real-time processing session
    """
    try:
        pipeline_config = PipelineConfig(**request["pipeline_config"])
        session_config = RealtimeSessionConfig(**request.get("session_config", {}))

        session_id = str(uuid.uuid4())

        # Convert pipeline config
        orchestration_config = await convert_pipeline_to_orchestration_format(pipeline_config)

        # Initialize session
        session = RealtimeSession(
            session_id=session_id,
            pipeline_id=pipeline_config.pipeline_id,
            status="running",
            metrics={
                "chunks_processed": 0,
                "average_latency": 0,
                "quality_score": 0,
            },
        )

        # Store session configuration
        active_sessions[session_id] = {
            "session": session,
            "pipeline_config": orchestration_config,
            "session_config": session_config,
            "websocket": None,
            "audio_coordinator": audio_coordinator,
        }

        logger.info(
            f"Started real-time session {session_id} for pipeline {pipeline_config.pipeline_id}"
        )

        return session

    except Exception as e:
        logger.error(f"Failed to start real-time session: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {e!s}",
        ) from e


@router.websocket("/realtime/{session_id}")
async def realtime_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time audio processing
    """
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.send_json({"type": "error", "error": "Session not found"})
        await websocket.close()
        return

    session_data = active_sessions[session_id]
    session_data["websocket"] = websocket

    logger.info(f"WebSocket connected for session {session_id}")

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "audio_chunk":
                # Process audio chunk
                await process_realtime_chunk(session_id, data["data"], websocket)

            elif data["type"] == "update_stage":
                # Update stage configuration
                await update_stage_config(
                    session_id, data["stage_id"], data["parameters"], websocket
                )

            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e!s}")
        await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        # Clean up session
        if session_id in active_sessions:
            active_sessions[session_id]["websocket"] = None


async def process_realtime_chunk(session_id: str, audio_data_b64: str, websocket: WebSocket):
    """
    Process a real-time audio chunk
    """
    try:
        session_data = active_sessions[session_id]
        audio_coordinator = session_data["audio_coordinator"]
        pipeline_config = session_data["pipeline_config"]

        # Decode audio data
        import base64

        audio_chunk = base64.b64decode(audio_data_b64)

        # Process chunk
        start_time = time.time()

        processed_result = await audio_coordinator.process_audio_chunk(
            audio_chunk, pipeline_config=pipeline_config
        )

        processing_time = (time.time() - start_time) * 1000

        # Update session metrics
        session = session_data["session"]
        session.metrics["chunks_processed"] += 1

        # Calculate running average latency
        current_avg = session.metrics.get("average_latency", 0)
        chunks_processed = session.metrics["chunks_processed"]
        session.metrics["average_latency"] = (
            current_avg * (chunks_processed - 1) + processing_time
        ) / chunks_processed

        # Send processed audio back
        if processed_result.get("processed_audio"):
            processed_audio_b64 = base64.b64encode(processed_result["processed_audio"]).decode(
                "utf-8"
            )
            await websocket.send_json({"type": "processed_audio", "audio": processed_audio_b64})

        # Send metrics update
        await websocket.send_json(
            {
                "type": "metrics",
                "metrics": {
                    "total_latency": processing_time,
                    "chunks_processed": session.metrics["chunks_processed"],
                    "average_latency": session.metrics["average_latency"],
                    "quality_metrics": {
                        "snr": processed_result.get("snr", 0),
                        "rms": processed_result.get("rms", -20.0),
                    },
                    "cpu_usage": processed_result.get("cpu_usage", 0),
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to process real-time chunk: {e!s}")
        await websocket.send_json({"type": "error", "error": f"Chunk processing failed: {e!s}"})


async def update_stage_config(
    session_id: str, stage_id: str, parameters: dict[str, Any], websocket: WebSocket
):
    """
    Update stage configuration in real-time
    """
    try:
        session_data = active_sessions[session_id]
        pipeline_config = session_data["pipeline_config"]

        # Update the configuration
        if stage_id in pipeline_config.get("stages", {}):
            pipeline_config["stages"][stage_id]["parameters"].update(parameters)

            # Apply updated configuration
            audio_coordinator = session_data["audio_coordinator"]
            await audio_coordinator.apply_pipeline_config(pipeline_config)

            await websocket.send_json(
                {"type": "config_updated", "stage_id": stage_id, "success": True}
            )
        else:
            await websocket.send_json({"type": "error", "error": f"Stage {stage_id} not found"})

    except Exception as e:
        logger.error(f"Failed to update stage config: {e!s}")
        await websocket.send_json({"type": "error", "error": f"Config update failed: {e!s}"})


async def convert_pipeline_to_orchestration_format(
    pipeline: PipelineConfig,
) -> dict[str, Any]:
    """
    Convert frontend pipeline format to orchestration service format
    """
    # Map frontend stage names to backend stage names
    stage_mapping = {
        "vad": "vad",
        "voice_filter": "voice_frequency_filter",
        "noise_reduction": "noise_reduction",
        "voice_enhancement": "voice_enhancement",
        "equalizer": "equalizer",
        "spectral_denoising": "spectral_denoising",
        "conventional_denoising": "conventional_denoising",
        "lufs_normalization": "lufs_normalization",
        "agc": "agc",
        "compression": "compression",
        "limiter": "limiter",
        "file_input": "input",
        "record_input": "input",
        "speaker_output": "output",
        "file_output": "output",
    }

    orchestration_config = {
        "pipeline_id": pipeline.pipeline_id,
        "name": pipeline.name,
        "stages": {},
        "connections": [
            {"source": conn.source_stage_id, "target": conn.target_stage_id}
            for conn in pipeline.connections
        ],
    }

    # Convert stages
    for frontend_name, stage_config in pipeline.stages.items():
        backend_name = stage_mapping.get(frontend_name, frontend_name)

        orchestration_config["stages"][backend_name] = {
            "enabled": stage_config.enabled,
            "gain_in": stage_config.gain_in,
            "gain_out": stage_config.gain_out,
            "parameters": stage_config.parameters,
        }

    return orchestration_config


@router.delete("/realtime/{session_id}")
async def stop_realtime_session(session_id: str):
    """
    Stop a real-time processing session
    """
    if session_id in active_sessions:
        session_data = active_sessions[session_id]

        # Close WebSocket if connected
        if session_data.get("websocket"):
            with contextlib.suppress(Exception):
                await session_data["websocket"].close()

        # Remove session
        del active_sessions[session_id]

        logger.info(f"Stopped real-time session {session_id}")
        return {"success": True, "message": f"Session {session_id} stopped"}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")


@router.get("/realtime/sessions")
async def list_active_sessions():
    """
    List all active real-time sessions
    """
    sessions = []
    for session_id, session_data in active_sessions.items():
        session = session_data["session"]
        sessions.append(
            {
                "session_id": session_id,
                "pipeline_id": session.pipeline_id,
                "status": session.status,
                "metrics": session.metrics,
                "connected": session_data.get("websocket") is not None,
            }
        )

    return {"active_sessions": sessions}
