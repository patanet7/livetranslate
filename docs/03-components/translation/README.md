# Translation Components

## Source Layout

- `modules/translation-service/src/api_server_fastapi.py`: FastAPI entrypoint.
- `modules/translation-service/src/translation_service.py`: primary translation service logic.
- `modules/translation-service/src/model_manager.py`: model/backends configuration and loading.
- `modules/translation-service/src/llama_translator.py`: Llama backend integration.
- `modules/translation-service/src/nllb_translator.py`: NLLB backend integration.
- `modules/translation-service/src/openai_compatible_translator.py`: OpenAI-compatible backend integration.
- `modules/translation-service/src/service_integration.py`: orchestration integration helpers.

## Service Documentation

- `modules/translation-service/README.md`
- `modules/translation-service/README-TRITON.md`
- `modules/translation-service/README-TRITON-SIMPLE.md`
