"""Audio processing components for LiveTranslate orchestration service."""

from .models import (
    AudioChunkMetadata,
    AudioChunkingConfig,
    ProcessingResult,
    ProcessingStatus,
    QualityMetrics,
    SpeakerCorrelation,
    SourceType,
    create_audio_chunk_metadata,
    get_default_chunking_config,
)

from .config import (
    AudioConfigurationManager,
    get_default_audio_processing_config,
    create_audio_config_manager,
)

# Note: These imports are commented out until the classes are implemented
# from .database_adapter import (
#     AudioDatabaseAdapter,
#     create_audio_database_adapter,
# )

# from .audio_processor import (
#     AudioPipelineProcessor,
#     create_audio_pipeline_processor,
# )

# from .chunk_manager import (
#     ChunkManager,
#     create_chunk_manager,
# )

# from .audio_coordinator import (
#     AudioCoordinator,
#     create_audio_coordinator,
# )

__all__ = [
    # Models
    "AudioChunkMetadata",
    "AudioChunkingConfig", 
    "ProcessingResult",
    "ProcessingStatus",
    "QualityMetrics",
    "SpeakerCorrelation",
    "SourceType",
    "create_audio_chunk_metadata",
    "get_default_chunking_config",
    
    # Configuration
    "AudioConfigurationManager",
    "get_default_audio_processing_config",
    "create_audio_config_manager",
    
    # Database (commented out until implemented)
    # "AudioDatabaseAdapter",
    # "create_audio_database_adapter",
    
    # Processing (commented out until implemented)
    # "AudioPipelineProcessor",
    # "create_audio_pipeline_processor",
    
    # Chunking (commented out until implemented)
    # "ChunkManager",
    # "create_chunk_manager",
    
    # Coordination (commented out until implemented)
    # "AudioCoordinator",
    # "create_audio_coordinator",
]