# Shared Libraries - Hardware Abstraction and Common Utilities

**Purpose**: Common utilities, hardware abstraction, and pipeline components for all services

## Library Overview

The Shared Libraries module provides:
- **Hardware Abstraction**: Unified NPU/GPU/CPU detection and management
- **Inference Clients**: Common interfaces for vLLM, Ollama, and Triton
- **Pipeline Components**: Reusable audio and text processing utilities
- **Configuration Management**: Centralized settings and environment handling
- **Performance Utilities**: Monitoring, logging, and optimization tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Shared Libraries                        │
│                    [Cross-Service Utilities]                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Hardware    │  │ Inference   │  │ Pipeline    │  │ Config  │ │
│  │ Abstraction │  │ Clients     │  │ Components  │  │ Mgmt    │ │
│  │ • NPU Det   │  │ • vLLM      │  │ • Audio     │  │ • Env   │ │
│  │ • GPU Det   │  │ • Ollama    │  │ • Text      │  │ • YAML  │ │
│  │ • CPU Det   │  │ • Triton    │  │ • Models    │  │ • JSON  │ │
│  │ • Fallback  │  │ • Base      │  │ • Utils     │  │ • Hot   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│                           ↓                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Performance │  │ Logging     │  │ Security    │  │ Testing │ │
│  │ • Metrics   │  │ • Struct    │  │ • Auth      │  │ • Mocks │ │
│  │ • Profiling │  │ • Tracing   │  │ • Validate  │  │ • Stubs │ │
│  │ • Monitor   │  │ • Export    │  │ • Crypto    │  │ • Utils │ │
│  │ • Optimize  │  │ • Analysis  │  │ • Keys      │  │ • Data  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Current Status

### ✅ Existing Components (Ready for Enhancement)
- **Basic Inference Clients** (`src/inference/`) - vLLM, Ollama, Triton clients
- **Pipeline Foundation** (`src/pipeline/`) - Real-time pipeline framework
- **Requirements** (`requirements.txt`) - Basic dependency management

### ⚠️ Hardware Abstraction Required
Need comprehensive hardware detection, management, and optimization utilities.

## Next Steps (HIGH PRIORITY)

### Phase 1: Hardware Abstraction Layer (Week 1)
1. **Create comprehensive hardware detection**:
   ```bash
   modules/shared/
   ├── src/
   │   ├── hardware/
   │   │   ├── __init__.py
   │   │   ├── detector.py              # Hardware detection
   │   │   ├── npu_manager.py           # Intel NPU management
   │   │   ├── gpu_manager.py           # NVIDIA GPU management
   │   │   ├── cpu_manager.py           # CPU optimization
   │   │   ├── device_selector.py       # Optimal device selection
   │   │   └── fallback_handler.py      # Hardware fallback logic
   │   ├── inference/
   │   │   ├── __init__.py
   │   │   ├── base_client.py           # Enhanced base client
   │   │   ├── vllm_client.py           # Enhanced vLLM client
   │   │   ├── ollama_client.py         # Enhanced Ollama client
   │   │   ├── triton_client.py         # Enhanced Triton client
   │   │   ├── factory.py               # Client factory pattern
   │   │   └── optimization.py          # Performance optimization
   │   ├── pipeline/
   │   │   ├── __init__.py
   │   │   ├── audio_pipeline.py        # Audio processing pipeline
   │   │   ├── text_pipeline.py         # Text processing pipeline
   │   │   ├── model_pipeline.py        # Model inference pipeline
   │   │   └── real_time_pipeline.py    # Enhanced real-time pipeline
   │   ├── config/
   │   │   ├── __init__.py
   │   │   ├── manager.py               # Configuration management
   │   │   ├── environment.py           # Environment handling
   │   │   ├── validation.py            # Config validation
   │   │   └── hot_reload.py            # Hot configuration reload
   │   ├── performance/
   │   │   ├── __init__.py
   │   │   ├── metrics.py               # Performance metrics
   │   │   ├── profiler.py              # Code profiling
   │   │   ├── monitor.py               # System monitoring
   │   │   └── optimizer.py             # Performance optimization
   │   ├── logging/
   │   │   ├── __init__.py
   │   │   ├── structured.py            # Structured logging
   │   │   ├── tracing.py               # Distributed tracing
   │   │   ├── exporters.py             # Log exporters
   │   │   └── analysis.py              # Log analysis
   │   ├── security/
   │   │   ├── __init__.py
   │   │   ├── authentication.py        # Auth utilities
   │   │   ├── validation.py            # Input validation
   │   │   ├── encryption.py            # Encryption utilities
   │   │   └── key_management.py        # Key management
   │   └── testing/
   │       ├── __init__.py
   │       ├── mocks.py                 # Mock utilities
   │       ├── stubs.py                 # Service stubs
   │       ├── fixtures.py              # Test fixtures
   │       └── data_generators.py       # Test data generation
   ```

2. **Implement hardware detection and management**:
   ```python
   # src/hardware/detector.py
   class HardwareDetector:
       def __init__(self):
           self.detected_hardware = {}
           self.capabilities = {}
       
       def detect_all_hardware(self) -> HardwareProfile:
           """Comprehensive hardware detection"""
           return HardwareProfile(
               npu=self.detect_npu(),
               gpu=self.detect_gpu(),
               cpu=self.detect_cpu()
           )
       
       def detect_npu(self) -> NPUInfo:
           """Detect Intel NPU capabilities"""
           # OpenVINO NPU detection
       
       def detect_gpu(self) -> GPUInfo:
           """Detect NVIDIA GPU capabilities"""
           # CUDA/ROCm GPU detection
       
       def detect_cpu(self) -> CPUInfo:
           """Detect CPU capabilities"""
           # CPU features, cores, AVX support
   ```

3. **Create device management system**:
   ```python
   # src/hardware/device_selector.py
   class DeviceSelector:
       def __init__(self):
           self.hardware_profile = HardwareDetector().detect_all_hardware()
           self.performance_history = {}
       
       def select_optimal_device(self, task_type: str, model_size: int) -> str:
           """Select optimal device based on task and hardware"""
           if task_type == "audio_processing" and self.hardware_profile.npu.available:
               return "npu"
           elif task_type == "translation" and self.hardware_profile.gpu.available:
               return "gpu"
           else:
               return "cpu"
       
       def get_fallback_chain(self, primary_device: str) -> List[str]:
           """Get fallback device chain"""
           return self.fallback_chains.get(primary_device, ["cpu"])
   ```

### Phase 2: Enhanced Inference Clients (Week 2)
1. **Create unified inference interface**:
   ```python
   # src/inference/factory.py
   class InferenceClientFactory:
       def __init__(self):
           self.device_selector = DeviceSelector()
           self.client_cache = {}
       
       def create_client(self, backend: str, model: str, device: str = None) -> BaseInferenceClient:
           """Create optimized inference client"""
           if device is None:
               device = self.device_selector.select_optimal_device("inference", get_model_size(model))
           
           client_key = f"{backend}_{model}_{device}"
           if client_key not in self.client_cache:
               self.client_cache[client_key] = self._create_client_instance(backend, model, device)
           
           return self.client_cache[client_key]
   ```

2. **Enhanced vLLM client with GPU optimization**:
   ```python
   # src/inference/vllm_client.py
   class OptimizedVLLMClient(BaseInferenceClient):
       def __init__(self, model: str, device: str = "gpu"):
           super().__init__(model, device)
           self.gpu_manager = GPUManager()
           self.memory_optimizer = MemoryOptimizer()
       
       async def initialize(self):
           """Initialize with GPU optimization"""
           self.gpu_memory_limit = self.gpu_manager.get_optimal_memory_limit()
           self.tensor_parallel_size = self.gpu_manager.get_optimal_parallel_size()
           
           self.llm = LLM(
               model=self.model,
               gpu_memory_utilization=self.gpu_memory_limit,
               tensor_parallel_size=self.tensor_parallel_size
           )
       
       async def infer(self, prompt: str, **kwargs) -> InferenceResult:
           """GPU-optimized inference"""
           with self.memory_optimizer.managed_inference():
               result = await self.llm.generate(prompt, **kwargs)
               return InferenceResult(
                   text=result.outputs[0].text,
                   tokens=result.usage.completion_tokens,
                   latency=result.metrics.inference_time,
                   device_used=self.device
               )
   ```

3. **Enhanced Ollama client with auto-configuration**:
   ```python
   # src/inference/ollama_client.py
   class OptimizedOllamaClient(BaseInferenceClient):
       def __init__(self, model: str, device: str = "auto"):
           super().__init__(model, device)
           self.auto_configurator = OllamaAutoConfigurator()
       
       async def initialize(self):
           """Auto-configure Ollama for optimal performance"""
           config = await self.auto_configurator.optimize_for_device(self.device)
           
           if self.device == "gpu":
               config.gpu_layers = self.auto_configurator.calculate_optimal_gpu_layers(self.model)
           
           await self.configure_ollama(config)
       
       async def auto_pull_model(self) -> bool:
           """Automatically pull model if not available"""
           if not await self.is_model_available():
               await self.pull_model()
               return True
           return False
   ```

### Phase 3: Pipeline Components and Utilities (Week 3)
1. **Advanced audio processing pipeline**:
   ```python
   # src/pipeline/audio_pipeline.py
   class AdvancedAudioPipeline:
       def __init__(self):
           self.device_selector = DeviceSelector()
           self.performance_monitor = PerformanceMonitor()
       
       def create_optimized_pipeline(self, config: AudioConfig) -> AudioPipeline:
           """Create hardware-optimized audio pipeline"""
           optimal_device = self.device_selector.select_optimal_device("audio", config.model_size)
           
           return AudioPipeline(
               preprocessor=self.create_preprocessor(optimal_device),
               model=self.load_model(config.model, optimal_device),
               postprocessor=self.create_postprocessor(),
               device=optimal_device
           )
       
       async def process_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[AudioResult]:
           """Process real-time audio stream with optimization"""
           async for chunk in audio_stream:
               with self.performance_monitor.measure("audio_processing"):
                   result = await self.pipeline.process(chunk)
                   yield result
   ```

2. **Text processing and quality validation**:
   ```python
   # src/pipeline/text_pipeline.py
   class TextProcessingPipeline:
       def __init__(self):
           self.quality_validator = TextQualityValidator()
           self.language_detector = LanguageDetector()
       
       async def process_text(self, text: str, context: ProcessingContext) -> TextResult:
           """Process text with quality validation"""
           # Language detection
           detected_lang = await self.language_detector.detect(text)
           
           # Quality validation
           quality_score = self.quality_validator.validate(text, detected_lang)
           
           # Processing based on context
           processed_text = await self.apply_processing(text, context)
           
           return TextResult(
               text=processed_text,
               language=detected_lang,
               quality_score=quality_score,
               processing_time=context.elapsed_time
           )
   ```

3. **Performance monitoring and optimization**:
   ```python
   # src/performance/monitor.py
   class PerformanceMonitor:
       def __init__(self):
           self.metrics_collector = MetricsCollector()
           self.performance_history = {}
       
       @contextmanager
       def measure(self, operation: str):
           """Measure operation performance"""
           start_time = time.time()
           start_memory = self.get_memory_usage()
           
           try:
               yield
           finally:
               duration = time.time() - start_time
               memory_delta = self.get_memory_usage() - start_memory
               
               self.record_performance(operation, duration, memory_delta)
       
       def get_optimization_suggestions(self) -> List[OptimizationSuggestion]:
           """AI-powered optimization suggestions"""
           return self.analyze_performance_patterns()
   ```

### Phase 4: Configuration and Testing Utilities (Week 4)
1. **Advanced configuration management**:
   ```python
   # src/config/manager.py
   class ConfigurationManager:
       def __init__(self):
           self.config_sources = {}
           self.validators = {}
           self.hot_reload_enabled = False
       
       def load_config(self, config_path: str) -> Configuration:
           """Load and validate configuration"""
           raw_config = self.load_raw_config(config_path)
           validated_config = self.validate_config(raw_config)
           hardware_config = self.optimize_for_hardware(validated_config)
           
           return Configuration(
               base=validated_config,
               hardware_optimized=hardware_config,
               environment=self.get_environment_overrides()
           )
       
       def enable_hot_reload(self, callback: Callable):
           """Enable hot configuration reload"""
           self.hot_reload_enabled = True
           self.watch_config_changes(callback)
   ```

2. **Comprehensive testing utilities**:
   ```python
   # src/testing/mocks.py
   class ServiceMocks:
       @staticmethod
       def create_audio_service_mock() -> AudioServiceMock:
           """Create audio service mock for testing"""
           return AudioServiceMock(
               transcription_latency=0.1,
               quality_score=0.95,
               device_simulation="npu"
           )
       
       @staticmethod
       def create_translation_service_mock() -> TranslationServiceMock:
           """Create translation service mock for testing"""
           return TranslationServiceMock(
               translation_quality=0.9,
               supported_languages=["en", "es", "fr", "de"],
               device_simulation="gpu"
           )
   
   # src/testing/fixtures.py
   class TestFixtures:
       @staticmethod
       def generate_audio_samples(duration: float, sample_rate: int = 16000) -> np.ndarray:
           """Generate test audio samples"""
           return np.random.normal(0, 0.1, int(duration * sample_rate)).astype(np.float32)
       
       @staticmethod
       def create_test_session() -> TestSession:
           """Create test session with mock services"""
           return TestSession(
               audio_service=ServiceMocks.create_audio_service_mock(),
               translation_service=ServiceMocks.create_translation_service_mock()
           )
   ```

## Comprehensive Testing

#### Unit Tests (`tests/unit/`)
```python
# tests/unit/test_hardware_detection.py
def test_hardware_detection():
    detector = HardwareDetector()
    profile = detector.detect_all_hardware()
    
    # Should detect at least CPU
    assert profile.cpu.available
    assert profile.cpu.cores > 0
    
    # GPU/NPU detection should not fail
    assert isinstance(profile.gpu.available, bool)
    assert isinstance(profile.npu.available, bool)

def test_device_selection():
    selector = DeviceSelector()
    
    # Test audio processing device selection
    device = selector.select_optimal_device("audio_processing", model_size=1024)
    assert device in ["npu", "gpu", "cpu"]
    
    # Test translation device selection
    device = selector.select_optimal_device("translation", model_size=4096)
    assert device in ["gpu", "cpu"]

# tests/unit/test_inference_clients.py
async def test_vllm_client():
    client = OptimizedVLLMClient("test-model", "cpu")  # Use CPU for testing
    await client.initialize()
    
    result = await client.infer("Test prompt")
    assert result.text is not None
    assert result.latency > 0
    assert result.device_used == "cpu"

async def test_client_factory():
    factory = InferenceClientFactory()
    
    # Test client creation
    client = factory.create_client("vllm", "test-model", "cpu")
    assert isinstance(client, OptimizedVLLMClient)
    
    # Test client caching
    client2 = factory.create_client("vllm", "test-model", "cpu")
    assert client is client2  # Should return cached instance

# tests/unit/test_pipeline_components.py
async def test_audio_pipeline():
    pipeline = AdvancedAudioPipeline()
    config = AudioConfig(model="whisper-base", sample_rate=16000)
    
    audio_pipeline = pipeline.create_optimized_pipeline(config)
    assert audio_pipeline.device in ["npu", "gpu", "cpu"]
    
    # Test with sample audio
    test_audio = TestFixtures.generate_audio_samples(1.0)  # 1 second
    result = await audio_pipeline.process(test_audio)
    assert result.transcription is not None

def test_performance_monitor():
    monitor = PerformanceMonitor()
    
    # Test performance measurement
    with monitor.measure("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    metrics = monitor.get_metrics("test_operation")
    assert metrics.duration >= 0.1
    assert metrics.memory_delta >= 0
```

#### Integration Tests (`tests/integration/`)
```python
# tests/integration/test_hardware_integration.py
def test_npu_gpu_fallback():
    """Test NPU to GPU to CPU fallback chain"""
    selector = DeviceSelector()
    
    # Simulate NPU failure
    selector.mark_device_unavailable("npu")
    
    device = selector.select_optimal_device("audio_processing", 1024)
    if selector.hardware_profile.gpu.available:
        assert device == "gpu"
    else:
        assert device == "cpu"

async def test_cross_service_pipeline():
    """Test pipeline components across services"""
    # Create pipeline with shared components
    audio_pipeline = AdvancedAudioPipeline()
    text_pipeline = TextProcessingPipeline()
    
    # Process audio → text → translation
    test_audio = TestFixtures.generate_audio_samples(2.0)
    audio_result = await audio_pipeline.process(test_audio)
    
    text_result = await text_pipeline.process_text(
        audio_result.transcription,
        ProcessingContext(task="transcription_cleanup")
    )
    
    assert text_result.quality_score > 0.7
    assert text_result.language is not None

# tests/integration/test_configuration_integration.py
def test_configuration_hot_reload():
    """Test hot configuration reload"""
    config_manager = ConfigurationManager()
    
    # Track configuration changes
    changes = []
    def track_changes(new_config):
        changes.append(new_config)
    
    config_manager.enable_hot_reload(track_changes)
    
    # Simulate configuration change
    config_manager.update_config({"new_setting": "value"})
    
    # Should trigger hot reload
    assert len(changes) > 0
    assert changes[-1].get("new_setting") == "value"
```

#### Performance Tests (`tests/performance/`)
```python
# tests/performance/test_inference_performance.py
async def test_vllm_throughput():
    """Test vLLM client throughput"""
    client = OptimizedVLLMClient("test-model", "gpu")
    await client.initialize()
    
    # Test batch processing
    prompts = ["Test prompt"] * 100
    start_time = time.time()
    
    results = await client.batch_infer(prompts)
    duration = time.time() - start_time
    
    throughput = len(results) / duration
    assert throughput > 10  # >10 inferences/second

def test_hardware_detection_performance():
    """Test hardware detection speed"""
    detector = HardwareDetector()
    
    start_time = time.time()
    profile = detector.detect_all_hardware()
    detection_time = time.time() - start_time
    
    assert detection_time < 1.0  # <1 second for detection
    assert profile is not None

# tests/performance/test_pipeline_performance.py
async def test_pipeline_latency():
    """Test end-to-end pipeline latency"""
    pipeline = AdvancedAudioPipeline()
    config = AudioConfig(model="whisper-small")
    
    audio_pipeline = pipeline.create_optimized_pipeline(config)
    
    # Test processing latency
    test_audio = TestFixtures.generate_audio_samples(1.0)
    start_time = time.time()
    
    result = await audio_pipeline.process(test_audio)
    latency = time.time() - start_time
    
    assert latency < 2.0  # <2 seconds for 1 second audio
    assert result.transcription is not None
```

## API Documentation

### Hardware Detection
```python
from modules.shared.hardware import HardwareDetector, DeviceSelector

# Detect available hardware
detector = HardwareDetector()
profile = detector.detect_all_hardware()

print(f"NPU Available: {profile.npu.available}")
print(f"GPU Available: {profile.gpu.available}")
print(f"CPU Cores: {profile.cpu.cores}")

# Select optimal device
selector = DeviceSelector()
device = selector.select_optimal_device("audio_processing", model_size=1024)
print(f"Optimal device: {device}")
```

### Inference Clients
```python
from modules.shared.inference import InferenceClientFactory

# Create optimized inference client
factory = InferenceClientFactory()
client = factory.create_client("vllm", "meta-llama/Llama-3.1-8B-Instruct", "gpu")

# Perform inference
result = await client.infer("Translate 'Hello' to Spanish")
print(f"Result: {result.text}")
print(f"Device used: {result.device_used}")
print(f"Latency: {result.latency}ms")
```

### Pipeline Components
```python
from modules.shared.pipeline import AdvancedAudioPipeline, TextProcessingPipeline

# Create optimized audio pipeline
audio_pipeline = AdvancedAudioPipeline()
config = AudioConfig(model="whisper-medium", device="npu")
pipeline = audio_pipeline.create_optimized_pipeline(config)

# Process audio stream
async for audio_chunk in audio_stream:
    result = await pipeline.process(audio_chunk)
    print(f"Transcription: {result.transcription}")
    print(f"Device: {result.device_used}")
```

### Performance Monitoring
```python
from modules.shared.performance import PerformanceMonitor

monitor = PerformanceMonitor()

# Measure operation performance
with monitor.measure("inference_operation"):
    result = await model.infer(prompt)

# Get performance metrics
metrics = monitor.get_metrics("inference_operation")
print(f"Average latency: {metrics.avg_duration}ms")
print(f"Memory usage: {metrics.avg_memory}MB")

# Get optimization suggestions
suggestions = monitor.get_optimization_suggestions()
for suggestion in suggestions:
    print(f"Suggestion: {suggestion.description}")
```

## Configuration

### Hardware Configuration (`config/hardware.yaml`)
```yaml
hardware:
  detection:
    auto_detect: true
    fallback_enabled: true
    performance_testing: true
  
  npu:
    enable: true
    optimization_level: 3
    memory_limit: 4096  # MB
    
  gpu:
    enable: true
    memory_utilization: 0.85
    multi_gpu_support: true
    
  cpu:
    enable: true
    optimization: "O3"
    thread_pool_size: auto

device_preferences:
  audio_processing: ["npu", "gpu", "cpu"]
  translation: ["gpu", "cpu"]
  general: ["gpu", "cpu"]
```

### Inference Configuration (`config/inference.yaml`)
```yaml
inference:
  clients:
    vllm:
      default_config:
        gpu_memory_utilization: 0.8
        tensor_parallel_size: 1
        max_model_len: 4096
    
    ollama:
      default_config:
        num_ctx: 4096
        num_gpu_layers: 35
        num_thread: auto
    
    triton:
      default_config:
        max_batch_size: 32
        instance_group:
          count: 1
          kind: KIND_GPU

optimization:
  enable_caching: true
  batch_optimization: true
  memory_pooling: true
  performance_profiling: true
```

This comprehensive shared libraries module provides the foundation for hardware-optimized, high-performance operation across all LiveTranslate services.