Check Python AST.........................................................Passed
Check docstring is first.................................................Failed
- hook id: check-docstring-first
- exit code: 1

modules/whisper-service/tests/integration/test_pytorch_real.py:442: Multiple module docstrings (first docstring on line 2).
modules/whisper-service/tests/integration/test_real_speech.py:182: Multiple module docstrings (first docstring on line 2).
modules/orchestration-service/src/audio/audio_coordinator_cache_integration.py:24: Multiple module docstrings (first docstring on line 1).
modules/whisper-service/tests/integration/test_openvino_real.py:344: Multiple module docstrings (first docstring on line 2).
modules/orchestration-service/tests/integration/test_audio_orchestration.py:1048: Multiple module docstrings (first docstring on line 1).

Check YAML syntax........................................................Failed
- hook id: check-yaml
- exit code: 1

while parsing a block mapping
  in "modules/frontend-service/.pre-commit-config.yaml", line 99, column 9
expected <block end>, but found ']'
  in "modules/frontend-service/.pre-commit-config.yaml", line 102, column 73
while parsing a block mapping
  in "modules/translation-service/.pre-commit-config.yaml", line 100, column 9
expected <block end>, but found ']'
  in "modules/translation-service/.pre-commit-config.yaml", line 103, column 68
while parsing a block mapping
  in "modules/orchestration-service/.pre-commit-config.yaml", line 119, column 9
expected <block end>, but found ']'
  in "modules/orchestration-service/.pre-commit-config.yaml", line 122, column 43

Check JSON syntax........................................................Passed
Check TOML syntax........................................................Passed
Check XML syntax.........................................................Passed
Fix end of files.........................................................Passed
Trim trailing whitespace.................................................Passed
Fix mixed line endings...................................................Passed
Check for merge conflicts................................................Passed
Protect main/master branches.............................................Failed
- hook id: no-commit-to-branch
- exit code: 1
Check for debug statements...............................................Passed
Check for large files....................................................Passed
Check builtin literals...................................................Passed
Check case conflicts.....................................................Passed
Check executables have shebangs..........................................Passed
Check shebang scripts are executable.....................................Failed
- hook id: check-shebang-scripts-are-executable
- exit code: 1

modules/bot-container/src/bot_main.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/src/bot_main.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/src/bot_main.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/src/orchestration_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/src/orchestration_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/src/orchestration_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/meeting-bot-service/scripts/build-production.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/meeting-bot-service/scripts/build-production.sh`
  If on Windows, you may also need to: `git add --chmod=+x modules/meeting-bot-service/scripts/build-production.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_caption_overlay.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_caption_overlay.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_caption_overlay.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/config_sync.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/config_sync.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/config_sync.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/equalizer_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/equalizer_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/equalizer_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/noise_reduction_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/noise_reduction_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/noise_reduction_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/voice_enhancement_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/voice_enhancement_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/voice_enhancement_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/time_correlation.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/time_correlation.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/time_correlation.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/routers/audio/websocket_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/routers/audio/websocket_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/routers/audio/websocket_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/socketio_whisper_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/socketio_whisper_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/socketio_whisper_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/websocket_whisper_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/websocket_whisper_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/websocket_whisper_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/conftest.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/conftest.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/conftest.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/unit/test_glossary_service.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/unit/test_glossary_service.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/unit/test_glossary_service.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_data_pipeline_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_data_pipeline_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_data_pipeline_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/scripts/discover_models.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/scripts/discover_models.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/scripts/discover_models.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/nllb_translator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/nllb_translator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/nllb_translator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/debug_test.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/debug_test.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/debug_test.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/download_chinese_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/download_chinese_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/download_chinese_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/audio_processor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/audio_processor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/audio_processor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/continuous_stream_processor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/continuous_stream_processor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/continuous_stream_processor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/language_id/smoother.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/language_id/smoother.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/language_id/smoother.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/utils/ring_buffer.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/utils/ring_buffer.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/utils/ring_buffer.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/start_npu.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/start_npu.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/start_npu.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/milestone2/test_real_code_switching.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/milestone2/test_real_code_switching.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/milestone2/test_real_code_switching.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_audio_optimization.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_audio_optimization.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_audio_optimization.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_whisper_service.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_whisper_service.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_whisper_service.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/test_config_flow.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/test_config_flow.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/test_config_flow.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/test_unit.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/test_unit.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/test_unit.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/test_websocket_stream_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/test_websocket_stream_server.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/test_websocket_stream_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/conftest.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/conftest.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/conftest.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_sustained_detector.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_sustained_detector.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_sustained_detector.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_task_parameter.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_task_parameter.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_task_parameter.py`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/quick_db_setup.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/quick_db_setup.sh`
  If on Windows, you may also need to: `git add --chmod=+x scripts/quick_db_setup.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

tests/system/test_error_handling.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x tests/system/test_error_handling.py`
  If on Windows, you may also need to: `git add --chmod=+x tests/system/test_error_handling.py`
  If it not supposed to be executable, double-check its shebang is wanted.

meeting-bot/start.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x meeting-bot/start.sh`
  If on Windows, you may also need to: `git add --chmod=+x meeting-bot/start.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/src/audio_capture.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/src/audio_capture.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/src/audio_capture.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/src/browser_automation.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/src/browser_automation.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/src/browser_automation.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/tests/unit/test_bot_main.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/tests/unit/test_bot_main.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/tests/unit/test_bot_main.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/meeting-bot-service/start.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/meeting-bot-service/start.sh`
  If on Windows, you may also need to: `git add --chmod=+x modules/meeting-bot-service/start.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/verify_enhanced_stages.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/verify_enhanced_stages.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/verify_enhanced_stages.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/examples/demo_streaming_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/examples/demo_streaming_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/examples/demo_streaming_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/voice_filter_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/voice_filter_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/voice_filter_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages_enhanced/limiter_enhanced.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages_enhanced/limiter_enhanced.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages_enhanced/limiter_enhanced.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/google_meet_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/google_meet_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/google_meet_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/config.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/config.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/config.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/database/processing_metrics.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/database/processing_metrics.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/database/processing_metrics.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/frontend/web_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/frontend/web_server.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/frontend/web_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/conftest.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/conftest.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/conftest.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/integration/test_glossary_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/integration/test_glossary_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/integration/test_glossary_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/unit/test_fireflies_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/unit/test_fireflies_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/unit/test_fireflies_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/run_comprehensive_audio_tests.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/run_comprehensive_audio_tests.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/run_comprehensive_audio_tests.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_bot_lifecycle.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_bot_lifecycle.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_bot_lifecycle.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/unit/test_enhanced_stages_smoke.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/unit/test_enhanced_stages_smoke.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/unit/test_enhanced_stages_smoke.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/openai_compatible_translator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/openai_compatible_translator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/openai_compatible_translator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/whisper_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/whisper_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/whisper_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/tests/integration/test_triton_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/tests/integration/test_triton_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/tests/integration/test_triton_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/audio/audio_utils.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/audio/audio_utils.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/audio/audio_utils.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/speaker_diarization.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/speaker_diarization.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/speaker_diarization.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/token_deduplicator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/token_deduplicator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/token_deduplicator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/vac_online_processor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/vac_online_processor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/vac_online_processor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/whisper_service.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/whisper_service.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/whisper_service.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/debug_hooks.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/debug_hooks.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/debug_hooks.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/fixtures.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/fixtures.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/fixtures.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_beam_search_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_beam_search_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_beam_search_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/test_performance_optimizations.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/test_performance_optimizations.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/test_performance_optimizations.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_token_deduplication.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_token_deduplication.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_token_deduplication.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/verify_mixed_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/verify_mixed_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/verify_mixed_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/SimulStreaming/whisper_streaming/whisper_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/SimulStreaming/whisper_streaming/whisper_server.py`
  If on Windows, you may also need to: `git add --chmod=+x reference/SimulStreaming/whisper_streaming/whisper_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/vexa/services/vexa-bot/core/build-browser-utils.js: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/vexa/services/vexa-bot/core/build-browser-utils.js`
  If on Windows, you may also need to: `git add --chmod=+x reference/vexa/services/vexa-bot/core/build-browser-utils.js`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/tests/integration/test_join_meeting.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/tests/integration/test_join_meeting.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/tests/integration/test_join_meeting.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/meeting-bot-service/xvfb-run-wrapper: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/meeting-bot-service/xvfb-run-wrapper`
  If on Windows, you may also need to: `git add --chmod=+x modules/meeting-bot-service/xvfb-run-wrapper`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/create_clean_mixed_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/create_clean_mixed_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/create_clean_mixed_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_obs_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_obs_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_obs_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/fix_imports.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/fix_imports.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/fix_imports.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/audio_processor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/audio_processor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/audio_processor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/database_adapter.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/database_adapter.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/database_adapter.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/models.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/models.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/models.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/start-dev.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/start-dev.ps1`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/start-dev.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/integration/test_chunk_manager_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/integration/test_chunk_manager_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/integration/test_chunk_manager_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/e2e/test_audio_streaming_e2e.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/e2e/test_audio_streaming_e2e.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/e2e/test_audio_streaming_e2e.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/integration/test_pipeline_dry_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/integration/test_pipeline_dry_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/integration/test_pipeline_dry_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fixtures/audio_test_data.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fixtures/audio_test_data.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fixtures/audio_test_data.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/model_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/model_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/model_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/service_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/service_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/service_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/service_integration_triton.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/service_integration_triton.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/service_integration_triton.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/beam_decoder.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/beam_decoder.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/beam_decoder.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/error_handler.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/error_handler.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/error_handler.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/language_id/decoder_utils.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/language_id/decoder_utils.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/language_id/decoder_utils.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/language_id/sustained_detector.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/language_id/sustained_detector.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/language_id/sustained_detector.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/sentence_segmenter.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/sentence_segmenter.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/sentence_segmenter.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/service_config.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/service_config.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/service_config.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/transcription/result_parser.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/transcription/result_parser.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/transcription/result_parser.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/utils/encoder_cache.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/utils/encoder_cache.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/utils/encoder_cache.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/accuracy/test_code_switching_accuracy.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/accuracy/test_code_switching_accuracy.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/accuracy/test_code_switching_accuracy.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_pytorch_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_pytorch_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_pytorch_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_silero_vad_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_silero_vad_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_silero_vad_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_warmup.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_warmup.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_warmup.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/SimulStreaming/whisper_streaming/line_packet.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/SimulStreaming/whisper_streaming/line_packet.py`
  If on Windows, you may also need to: `git add --chmod=+x reference/SimulStreaming/whisper_streaming/line_packet.py`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/start-database.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/start-database.ps1`
  If on Windows, you may also need to: `git add --chmod=+x scripts/start-database.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

start-development.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x start-development.ps1`
  If on Windows, you may also need to: `git add --chmod=+x start-development.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

start_whisper_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x start_whisper_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x start_whisper_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

meeting-bot/scripts/test-production-build.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x meeting-bot/scripts/test-production-build.sh`
  If on Windows, you may also need to: `git add --chmod=+x meeting-bot/scripts/test-production-build.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/src/redis_subscriber.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/src/redis_subscriber.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/src/redis_subscriber.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/tests/unit/test_orchestration_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/tests/unit/test_orchestration_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/tests/unit/test_orchestration_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_enhanced_default.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_enhanced_default.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_enhanced_default.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/audio_coordinator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/audio_coordinator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/audio_coordinator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/vad_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/vad_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/vad_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages_enhanced/__init__.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages_enhanced/__init__.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages_enhanced/__init__.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/audio_capture.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/audio_capture.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/audio_capture.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/database/unified_bot_session_repository.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/database/unified_bot_session_repository.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/database/unified_bot_session_repository.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/monitoring/health_monitor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/monitoring/health_monitor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/monitoring/health_monitor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/utils/dependency_check.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/utils/dependency_check.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/utils/dependency_check.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/model_downloader.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/model_downloader.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/model_downloader.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/translation_service.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/translation_service.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/translation_service.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/vllm_server_simple.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/vllm_server_simple.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/vllm_server_simple.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/tests/integration/test_vllm_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/tests/integration/test_vllm_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/tests/integration/test_vllm_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/debug_mixed_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/debug_mixed_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/debug_mixed_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/scripts/benchmark_optimizations.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/scripts/benchmark_optimizations.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/scripts/benchmark_optimizations.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/config/config_loader.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/config/config_loader.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/config/config_loader.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/pipeline_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/pipeline_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/pipeline_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/session/session_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/session/session_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/session/session_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/transcription/text_analysis.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/transcription/text_analysis.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/transcription/text_analysis.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/type_definitions.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/type_definitions.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/type_definitions.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/utils/performance_metrics.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/utils/performance_metrics.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/utils/performance_metrics.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/property/test_invariants.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/property/test_invariants.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/property/test_invariants.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_whisper_lid_probe.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_whisper_lid_probe.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_whisper_lid_probe.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_whisper_service_helpers.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_whisper_service_helpers.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_whisper_service_helpers.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/vexa/services/WhisperLive/scripts/setup.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/vexa/services/WhisperLive/scripts/setup.sh`
  If on Windows, you may also need to: `git add --chmod=+x reference/vexa/services/WhisperLive/scripts/setup.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/test-integration.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/test-integration.ps1`
  If on Windows, you may also need to: `git add --chmod=+x scripts/test-integration.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/tests/integration/test_simple_join.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/tests/integration/test_simple_join.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/tests/integration/test_simple_join.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/frontend-service/fix-ts6133.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/frontend-service/fix-ts6133.sh`
  If on Windows, you may also need to: `git add --chmod=+x modules/frontend-service/fix-ts6133.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_audio_pipeline.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_audio_pipeline.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_audio_pipeline.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_obs_simple.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_obs_simple.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_obs_simple.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/scripts/deploy-monitoring.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/scripts/deploy-monitoring.sh`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/scripts/deploy-monitoring.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/chunk_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/chunk_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/chunk_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/__init__.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/__init__.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/__init__.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages_enhanced/lufs_normalization_enhanced.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages_enhanced/lufs_normalization_enhanced.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages_enhanced/lufs_normalization_enhanced.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/whisper_compatibility.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/whisper_compatibility.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/whisper_compatibility.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/bot_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/bot_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/bot_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/bot_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/bot_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/bot_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/caption_processor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/caption_processor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/caption_processor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/main_fastapi.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/main_fastapi.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/main_fastapi.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/pipeline/data_pipeline.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/pipeline/data_pipeline.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/pipeline/data_pipeline.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/conftest.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/conftest.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/conftest.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/unit/test_audio_models_basic.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/unit/test_audio_models_basic.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/unit/test_audio_models_basic.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/run_fireflies_tests.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/run_fireflies_tests.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/run_fireflies_tests.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/unit/test_pipeline_fixes.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/unit/test_pipeline_fixes.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/unit/test_pipeline_fixes.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/scripts/start-triton-translation.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/scripts/start-triton-translation.sh`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/scripts/start-triton-translation.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/tests/quick_model_switch_test.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/tests/quick_model_switch_test.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/tests/quick_model_switch_test.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/connection_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/connection_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/connection_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/logging_utils.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/logging_utils.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/logging_utils.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/message_router.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/message_router.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/message_router.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/silero_vad_iterator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/silero_vad_iterator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/silero_vad_iterator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/simple_auth.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/simple_auth.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/simple_auth.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/token_buffer.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/token_buffer.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/token_buffer.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/websocket_stream_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/websocket_stream_server.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/websocket_stream_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/check_npu_standalone.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/check_npu_standalone.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/check_npu_standalone.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_jfk_domain_prompts.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_jfk_domain_prompts.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_jfk_domain_prompts.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_jfk_via_orchestration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_jfk_via_orchestration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_jfk_via_orchestration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_pytorch_real.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_pytorch_real.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_pytorch_real.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_real_speech.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_real_speech.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_real_speech.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/smoke/test_jfk_direct.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/smoke/test_jfk_direct.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/smoke/test_jfk_direct.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/SimulStreaming/simulstreaming_whisper_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/SimulStreaming/simulstreaming_whisper_server.py`
  If on Windows, you may also need to: `git add --chmod=+x reference/SimulStreaming/simulstreaming_whisper_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/SimulStreaming/whisper_streaming/whisper_online_main.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/SimulStreaming/whisper_streaming/whisper_online_main.py`
  If on Windows, you may also need to: `git add --chmod=+x reference/SimulStreaming/whisper_streaming/whisper_online_main.py`
  If it not supposed to be executable, double-check its shebang is wanted.

meeting-bot/xvfb-run-wrapper: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x meeting-bot/xvfb-run-wrapper`
  If on Windows, you may also need to: `git add --chmod=+x meeting-bot/xvfb-run-wrapper`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/meeting-bot-service/scripts/test-production-build.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/meeting-bot-service/scripts/test-production-build.sh`
  If on Windows, you may also need to: `git add --chmod=+x modules/meeting-bot-service/scripts/test-production-build.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/tcp_audio_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/tcp_audio_server.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/tcp_audio_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_enhanced_stages_instantiation.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_enhanced_stages_instantiation.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_enhanced_stages_instantiation.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_runtime_switching.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_runtime_switching.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_runtime_switching.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/config.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/config.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/config.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/conventional_denoising_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/conventional_denoising_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/conventional_denoising_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/spectral_denoising_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/spectral_denoising_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/spectral_denoising_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/dependencies.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/dependencies.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/dependencies.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/mocks/fireflies_mock_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/mocks/fireflies_mock_server.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/mocks/fireflies_mock_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/unit/test_sentence_aggregator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/unit/test_sentence_aggregator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/unit/test_sentence_aggregator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/integration/test_ws_connection.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/integration/test_ws_connection.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/integration/test_ws_connection.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_docker_manager_simple.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_docker_manager_simple.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_docker_manager_simple.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/llama_translator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/llama_translator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/llama_translator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/tests/test_model_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/tests/test_model_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/tests/test_model_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/legacy/api_server_before_phase2.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/legacy/api_server_before_phase2.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/legacy/api_server_before_phase2.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/api_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/api_server.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/api_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/models/openvino_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/models/openvino_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/models/openvino_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/stream_session_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/stream_session_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/stream_session_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/text_language_detector.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/text_language_detector.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/text_language_detector.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/transcription/domain_prompt_helper.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/transcription/domain_prompt_helper.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/transcription/domain_prompt_helper.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_cif_word_boundaries_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_cif_word_boundaries_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_cif_word_boundaries_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_jfk_streaming_simulation.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_jfk_streaming_simulation.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_jfk_streaming_simulation.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_utf8_fix_real_data.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_utf8_fix_real_data.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_utf8_fix_real_data.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/stress/test_stress.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/stress/test_stress.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/stress/test_stress.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/test_utils.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/test_utils.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/test_utils.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_vad_enhancement.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_vad_enhancement.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_vad_enhancement.py`
  If it not supposed to be executable, double-check its shebang is wanted.

tests/system/test_model_selection.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x tests/system/test_model_selection.py`
  If on Windows, you may also need to: `git add --chmod=+x tests/system/test_model_selection.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/src/google_meet_automation.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/src/google_meet_automation.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/src/google_meet_automation.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/create_mixed_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/create_mixed_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/create_mixed_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/start_backend.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/start_backend.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/start_backend.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/dashboard/real_time_dashboard.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/dashboard/real_time_dashboard.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/dashboard/real_time_dashboard.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/routers/bot/bot_docker_callbacks.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/routers/bot/bot_docker_callbacks.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/routers/bot/bot_docker_callbacks.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/e2e/test_meeting_bot_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/e2e/test_meeting_bot_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/e2e/test_meeting_bot_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/e2e/test_multiple_meeting_codes.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/e2e/test_multiple_meeting_codes.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/e2e/test_multiple_meeting_codes.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/unit/test_fireflies_router.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/unit/test_fireflies_router.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/unit/test_fireflies_router.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/integration/test_complete_audio_flow.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/integration/test_complete_audio_flow.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/integration/test_complete_audio_flow.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_audio_capture.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_audio_capture.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_audio_capture.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_translation_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_translation_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_translation_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/start-translation-service.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/start-translation-service.ps1`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/start-translation-service.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/create_warmup_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/create_warmup_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/create_warmup_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/eow_detection.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/eow_detection.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/eow_detection.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/language_id/lid_detector.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/language_id/lid_detector.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/language_id/lid_detector.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/segment_timestamper.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/segment_timestamper.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/segment_timestamper.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/sliding_lid_detector.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/sliding_lid_detector.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/sliding_lid_detector.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/stability_tracker.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/stability_tracker.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/stability_tracker.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/test_chinese_only.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/test_chinese_only.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/test_chinese_only.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_domain_prompts.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_domain_prompts.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_domain_prompts.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/milestone1/test_baseline_transcription.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/milestone1/test_baseline_transcription.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/milestone1/test_baseline_transcription.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_kv_cache_masking.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_kv_cache_masking.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_kv_cache_masking.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_token_buffer.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_token_buffer.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_token_buffer.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/vexa/services/WhisperLive/scripts/build_whisper_tensorrt.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/vexa/services/WhisperLive/scripts/build_whisper_tensorrt.sh`
  If on Windows, you may also need to: `git add --chmod=+x reference/vexa/services/WhisperLive/scripts/build_whisper_tensorrt.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/src/manual_login_helper.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/src/manual_login_helper.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/src/manual_login_helper.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_ab_comparison.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_ab_comparison.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_ab_comparison.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/speaker_correlator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/speaker_correlator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/speaker_correlator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/database/base.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/database/base.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/database/base.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/gateway/api_gateway.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/gateway/api_gateway.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/gateway/api_gateway.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/routers/bot/bot_docker_management.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/routers/bot/bot_docker_management.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/routers/bot/bot_docker_management.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/routers/data_query.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/routers/data_query.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/routers/data_query.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/start-backend-poetry.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/start-backend-poetry.ps1`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/start-backend-poetry.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/start-dev-simple.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/start-dev-simple.ps1`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/start-dev-simple.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/integration/test_audio_coordinator_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/integration/test_audio_coordinator_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/integration/test_audio_coordinator_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/performance/test_audio_performance.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/performance/test_audio_performance.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/performance/test_audio_performance.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/unit/test_fireflies_models.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/unit/test_fireflies_models.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/unit/test_fireflies_models.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_websocket_whisper_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_websocket_whisper_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_websocket_whisper_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/api_server_fastapi.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/api_server_fastapi.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/api_server_fastapi.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/tests/integration/test_model_switching_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/tests/integration/test_model_switching_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/tests/integration/test_model_switching_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/legacy/api_server_WORKING.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/legacy/api_server_WORKING.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/legacy/api_server_WORKING.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/legacy/vac_online_processor_WORKING.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/legacy/vac_online_processor_WORKING.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/legacy/vac_online_processor_WORKING.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/alignatt_decoder.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/alignatt_decoder.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/alignatt_decoder.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/utils/__init__.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/utils/__init__.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/utils/__init__.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_openvino_real.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_openvino_real.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_openvino_real.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/run_tests.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/run_tests.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/run_tests.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/unit/test_vad.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/unit/test_vad.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/unit/test_vad.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/vexa/services/WhisperLive/entrypoint.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/vexa/services/WhisperLive/entrypoint.sh`
  If on Windows, you may also need to: `git add --chmod=+x reference/vexa/services/WhisperLive/entrypoint.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/vexa/services/WhisperLive/healthcheck.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/vexa/services/WhisperLive/healthcheck.sh`
  If on Windows, you may also need to: `git add --chmod=+x reference/vexa/services/WhisperLive/healthcheck.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/export_openapi.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/export_openapi.py`
  If on Windows, you may also need to: `git add --chmod=+x scripts/export_openapi.py`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/start-database.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/start-database.sh`
  If on Windows, you may also need to: `git add --chmod=+x scripts/start-database.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

tests/system/test_multipart.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x tests/system/test_multipart.py`
  If on Windows, you may also need to: `git add --chmod=+x tests/system/test_multipart.py`
  If it not supposed to be executable, double-check its shebang is wanted.

meeting-bot/scripts/build-production.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x meeting-bot/scripts/build-production.sh`
  If on Windows, you may also need to: `git add --chmod=+x meeting-bot/scripts/build-production.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/bot-container/tests/integration/test_login.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/bot-container/tests/integration/test_login.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/bot-container/tests/integration/test_login.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/frontend-service/start-frontend.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/frontend-service/start-frontend.ps1`
  If on Windows, you may also need to: `git add --chmod=+x modules/frontend-service/start-frontend.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/debug_health_monitor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/debug_health_monitor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/debug_health_monitor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stage_components.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stage_components.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stage_components.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages/agc_stage.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages/agc_stage.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages/agc_stage.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/bot_lifecycle_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/bot_lifecycle_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/bot_lifecycle_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/docker_bot_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/docker_bot_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/docker_bot_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/bot/virtual_webcam.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/bot/virtual_webcam.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/bot/virtual_webcam.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/utils/logger.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/utils/logger.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/utils/logger.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/unit/test_audio_models.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/unit/test_audio_models.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/unit/test_audio_models.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/integration/test_mock_server_api_contract.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/integration/test_mock_server_api_contract.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/integration/test_mock_server_api_contract.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/integration/test_pipeline_production_readiness.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/integration/test_pipeline_production_readiness.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/integration/test_pipeline_production_readiness.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_whisper_websocket_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_whisper_websocket_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_whisper_websocket_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/shared/src/pipeline/real_time_pipeline.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/shared/src/pipeline/real_time_pipeline.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/shared/src/pipeline/real_time_pipeline.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/src/api_server.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/src/api_server.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/src/api_server.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/legacy/whisper_service_before_phase2.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/legacy/whisper_service_before_phase2.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/legacy/whisper_service_before_phase2.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/audio/vad_processor.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/audio/vad_processor.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/audio/vad_processor.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/domain_prompt_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/domain_prompt_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/domain_prompt_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/main.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/main.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/main.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/models/base_model.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/models/base_model.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/models/base_model.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/orchestration/response_formatter.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/orchestration/response_formatter.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/orchestration/response_formatter.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/utf8_boundary_fixer.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/utf8_boundary_fixer.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/utf8_boundary_fixer.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/__init__.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/__init__.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/__init__.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_adaptive_chunking_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_adaptive_chunking_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_adaptive_chunking_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_jfk_basic_orchestration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_jfk_basic_orchestration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_jfk_basic_orchestration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/stress/test_long_session.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/stress/test_long_session.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/stress/test_long_session.py`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/deploy.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/deploy.py`
  If on Windows, you may also need to: `git add --chmod=+x scripts/deploy.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_audio_upload.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_audio_upload.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_audio_upload.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/docs/scripts/test_socketio_whisper_client.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/docs/scripts/test_socketio_whisper_client.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/docs/scripts/test_socketio_whisper_client.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/examples/demo_virtual_webcam_live.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/examples/demo_virtual_webcam_live.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/examples/demo_virtual_webcam_live.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/stages_enhanced/compression_enhanced.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/stages_enhanced/compression_enhanced.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/stages_enhanced/compression_enhanced.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/audio/timing_coordinator.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/audio/timing_coordinator.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/audio/timing_coordinator.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/database/bot_session_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/database/bot_session_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/database/bot_session_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/main.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/main.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/main.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/src/websocket_frontend_handler.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/src/websocket_frontend_handler.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/src/websocket_frontend_handler.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/start-backend.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/start-backend.ps1`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/start-backend.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/audio/run_audio_tests.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/audio/run_audio_tests.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/audio/run_audio_tests.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/fireflies/integration/test_fireflies_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/fireflies/integration/test_fireflies_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/fireflies/integration/test_fireflies_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/orchestration-service/tests/test_docker_bot_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/orchestration-service/tests/test_docker_bot_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/orchestration-service/tests/test_docker_bot_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/translation-service/start-local.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/translation-service/start-local.sh`
  If on Windows, you may also need to: `git add --chmod=+x modules/translation-service/start-local.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/legacy/whisper_service_WORKING.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/legacy/whisper_service_WORKING.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/legacy/whisper_service_WORKING.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/heartbeat_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/heartbeat_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/heartbeat_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/models/model_factory.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/models/model_factory.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/models/model_factory.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/models/pytorch_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/models/pytorch_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/models/pytorch_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/reconnection_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/reconnection_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/reconnection_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/session_restart/session_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/session_restart/session_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/session_restart/session_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/transcript_manager.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/transcript_manager.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/transcript_manager.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/transcription/request_models.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/transcription/request_models.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/transcription/request_models.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/vad_detector.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/vad_detector.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/vad_detector.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/src/vad_helpers.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/src/vad_helpers.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/src/vad_helpers.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/benchmarks/test_latency.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/benchmarks/test_latency.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/benchmarks/test_latency.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_alignatt_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_alignatt_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_alignatt_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_domain_prompt_integration.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_domain_prompt_integration.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_domain_prompt_integration.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/test_streaming_stability.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/test_streaming_stability.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/test_streaming_stability.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/integration/tests/integration/test_mixed_direct.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/integration/tests/integration/test_mixed_direct.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/integration/tests/integration/test_mixed_direct.py`
  If it not supposed to be executable, double-check its shebang is wanted.

modules/whisper-service/tests/utils.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x modules/whisper-service/tests/utils.py`
  If on Windows, you may also need to: `git add --chmod=+x modules/whisper-service/tests/utils.py`
  If it not supposed to be executable, double-check its shebang is wanted.

reference/vexa/services/vexa-bot/core/entrypoint.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x reference/vexa/services/vexa-bot/core/entrypoint.sh`
  If on Windows, you may also need to: `git add --chmod=+x reference/vexa/services/vexa-bot/core/entrypoint.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/deploy-monitoring.ps1: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/deploy-monitoring.ps1`
  If on Windows, you may also need to: `git add --chmod=+x scripts/deploy-monitoring.ps1`
  If it not supposed to be executable, double-check its shebang is wanted.

scripts/load-secrets.sh: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x scripts/load-secrets.sh`
  If on Windows, you may also need to: `git add --chmod=+x scripts/load-secrets.sh`
  If it not supposed to be executable, double-check its shebang is wanted.

tests/system/test_real_audio.py: has a shebang but is not marked executable!
  If it is supposed to be executable, try: `chmod +x tests/system/test_real_audio.py`
  If on Windows, you may also need to: `git add --chmod=+x tests/system/test_real_audio.py`
  If it not supposed to be executable, double-check its shebang is wanted.

Check symlinks.......................................(no files to check)Skipped
Check destroyed symlinks.................................................Passed
Fix byte order marker....................................................Passed
Tests should be named test_*.py..........................................Failed
- hook id: name-tests-test
- exit code: 1

modules/orchestration-service/tests/audio/run_audio_tests.py does not match pattern "test_.*\.py"
modules/whisper-service/tests/fixtures.py does not match pattern "test_.*\.py"
modules/orchestration-service/tests/fireflies/mocks/fireflies_mock_server.py does not match pattern "test_.*\.py"
modules/translation-service/tests/quick_model_switch_test.py does not match pattern "test_.*\.py"
modules/whisper-service/tests/debug_hooks.py does not match pattern "test_.*\.py"
modules/orchestration-service/tests/fixtures/audio_test_data.py does not match pattern "test_.*\.py"
modules/orchestration-service/tests/fireflies/run_fireflies_tests.py does not match pattern "test_.*\.py"
modules/whisper-service/tests/run_tests.py does not match pattern "test_.*\.py"
modules/whisper-service/tests/utils.py does not match pattern "test_.*\.py"
modules/orchestration-service/tests/run_comprehensive_audio_tests.py does not match pattern "test_.*\.py"
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py does not match pattern "test_.*\.py"
modules/whisper-service/tests/integration/check_npu_standalone.py does not match pattern "test_.*\.py"

Fix requirements.txt.....................................................Passed
Ruff linter (all Python).................................................Failed
- hook id: ruff
- exit code: 1

modules/orchestration-service/src/audio/config.py:801:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
799 |           for key, value in config_dict.items():
800 |               if hasattr(self, key):
801 |                   if isinstance(
    |  ____________________^
802 | |                     getattr(self, key),
803 | |                     (
804 | |                         VADConfig,
805 | |                         VoiceFilterConfig,
806 | |                         NoiseReductionConfig,
807 | |                         VoiceEnhancementConfig,
808 | |                         EqualizerConfig,
809 | |                         SpectralDenoisingConfig,
810 | |                         ConventionalDenoisingConfig,
811 | |                         LUFSNormalizationConfig,
812 | |                         AGCConfig,
813 | |                         CompressionConfig,
814 | |                         LimiterConfig,
815 | |                         QualityConfig,
816 | |                     ),
817 | |                 ):
    | |_________________^ UP038
818 |                       # Update nested config objects
819 |                       current_config = getattr(self, key)
    |
    = help: Convert to `X | Y`

modules/orchestration-service/src/audio/config.py:1052:18: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
     |
1050 |                 # Handle nested objects
1051 |                 return {k: convert_value(v) for k, v in value.__dict__.items()}
1052 |             elif isinstance(value, (list, tuple)):
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
1053 |                 return [convert_value(item) for item in value]
1054 |             elif isinstance(value, dict):
     |
     = help: Convert to `X | Y`

modules/orchestration-service/src/audio/config_sync.py:180:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
178 |         if "inference_interval" in config:
179 |             interval = config["inference_interval"]
180 |             if not isinstance(interval, (int, float)) or interval < 0.5 or interval > 30.0:
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
181 |                 result["errors"].append(
182 |                     f"Invalid inference_interval: {interval}. Must be between 0.5 and 30.0 seconds"
    |
    = help: Convert to `X | Y`

modules/orchestration-service/src/audio/config_sync.py:187:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
185 |         if "buffer_duration" in config:
186 |             buffer = config["buffer_duration"]
187 |             if not isinstance(buffer, (int, float)) or buffer < 1.0 or buffer > 60.0:
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
188 |                 result["errors"].append(
189 |                     f"Invalid buffer_duration: {buffer}. Must be between 1.0 and 60.0 seconds"
    |
    = help: Convert to `X | Y`

modules/orchestration-service/src/audio/config_sync.py:208:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
206 |         if "chunk_duration" in config:
207 |             duration = config["chunk_duration"]
208 |             if not isinstance(duration, (int, float)) or duration < 1.0 or duration > 30.0:
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
209 |                 result["errors"].append(
210 |                     f"Invalid chunk_duration: {duration}. Must be between 1.0 and 30.0 seconds"
    |
    = help: Convert to `X | Y`

modules/orchestration-service/src/audio/config_sync.py:216:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
214 |         if "overlap_duration" in config:
215 |             overlap = config["overlap_duration"]
216 |             if not isinstance(overlap, (int, float)) or overlap < 0.0 or overlap > 5.0:
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
217 |                 result["errors"].append(
218 |                     f"Invalid overlap_duration: {overlap}. Must be between 0.0 and 5.0 seconds"
    |
    = help: Convert to `X | Y`

modules/orchestration-service/src/audio/config_sync.py:245:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
243 |         if "temperature" in config:
244 |             temp = config["temperature"]
245 |             if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
246 |                 result["errors"].append(f"Invalid temperature: {temp}. Must be between 0.0 and 2.0")
    |
    = help: Convert to `X | Y`

modules/orchestration-service/src/gateway/api_gateway.py:484:21: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
482 |                   # Don't retry on certain errors
483 |                   if (
484 |                       isinstance(
    |  _____________________^
485 | |                         e,
486 | |                         (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
487 | |                     )
    | |_____________________^ UP038
488 |                       and attempt < self.retries - 1
489 |                   ):
    |
    = help: Convert to `X | Y`

modules/seamless/src/streaming_st.py:14:19: RUF009 Do not perform function call `os.getenv` in dataclass defaults
   |
12 |     target_lang: str = "eng"  # English
13 |     sample_rate: int = 16000
14 |     device: str = os.getenv("DEVICE", "cpu")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^ RUF009
15 |     model_name: str = os.getenv("SEAMLESS_MODEL", "facebook/seamless-m4t-v2-large")
16 |     max_seconds_per_inference: float = 3.0  # window size for partial inference
   |

modules/seamless/src/streaming_st.py:15:23: RUF009 Do not perform function call `os.getenv` in dataclass defaults
   |
13 |     sample_rate: int = 16000
14 |     device: str = os.getenv("DEVICE", "cpu")
15 |     model_name: str = os.getenv("SEAMLESS_MODEL", "facebook/seamless-m4t-v2-large")
   |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RUF009
16 |     max_seconds_per_inference: float = 3.0  # window size for partial inference
   |

modules/shared/src/audio/audio_validator.py:308:16: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
306 |         try:
307 |             # Convert to numpy array if needed
308 |             if isinstance(audio_data, (bytes, str)):
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
309 |                 if isinstance(audio_data, str):
310 |                     audio_array, sr = librosa.load(audio_data, sr=None)
    |
    = help: Convert to `X | Y`

modules/shared/src/audio/audio_validator.py:448:16: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
446 |         try:
447 |             # Convert to numpy array if needed
448 |             if isinstance(audio_data, (bytes, str)):
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
449 |                 if isinstance(audio_data, str):
450 |                     audio_array, sr = librosa.load(audio_data, sr=None)
    |
    = help: Convert to `X | Y`

modules/whisper-service/src/simul_whisper/whisper/trans_nopad.py:156:41: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
155 |     def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
156 |         temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
    |                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
157 |         decode_result = None
    |
    = help: Convert to `X | Y`

modules/whisper-service/src/simul_whisper/whisper/transcribe.py:148:41: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
147 |     def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
148 |         temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
    |                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
149 |         decode_result = None
    |
    = help: Convert to `X | Y`

modules/whisper-service/src/utils/audio_errors.py:265:16: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
264 |     def can_recover(self, error: WhisperProcessingBaseError) -> bool:
265 |         return isinstance(error, (ModelLoadingError, ModelInferenceError))
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
266 | 
267 |     def recover(self, error: WhisperProcessingBaseError, context: dict[str, Any]) -> dict[str, Any]:
    |
    = help: Convert to `X | Y`

modules/whisper-service/tests/integration/test_integration.py:160:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
159 |             assert data["status"] in ["healthy", "ok"]
160 |             assert isinstance(data["timestamp"], (int, float))
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
161 | 
162 |     @pytest.mark.asyncio
    |
    = help: Convert to `X | Y`

modules/whisper-service/tests/integration/test_openvino_real.py:106:16: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
104 |         # Verify we got actual results
105 |         assert result is not None
106 |         assert isinstance(result, (str, dict))
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
107 | 
108 |         text = result if isinstance(result, str) else result.get("text", "")
    |
    = help: Convert to `X | Y`

modules/whisper-service/tests/property/test_invariants.py:214:28: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
212 |                   for key in result:
213 |                       assert key in ["start", "end"], f"Invalid key in VAD result: {key}"
214 |                       assert isinstance(
    |  ____________________________^
215 | |                         result[key], (int, float)
216 | |                     ), f"Invalid value type for {key}: {type(result[key])}"
    | |_____________________^ UP038
217 |                       assert result[key] >= 0, f"Negative timestamp: {result[key]}"
    |
    = help: Convert to `X | Y`

modules/whisper-service/tests/unit/test_vad.py:211:20: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
210 |         if result is not None and "start" in result:
211 |             assert isinstance(result["start"], (int, float))
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
212 |             assert result["start"] >= 0
213 |             logger.info(f"✅ START event format correct: {result}")
    |
    = help: Convert to `X | Y`

modules/whisper-service/tests/unit/test_vad.py:229:24: UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
228 |             if result is not None and "end" in result:
229 |                 assert isinstance(result["end"], (int, float))
    |                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
230 |                 assert result["end"] >= 0
231 |                 end_detected = True
    |
    = help: Convert to `X | Y`

Found 20 errors.
No fixes available (18 hidden fixes can be enabled with the `--unsafe-fixes` option).

Ruff formatter (all Python)..............................................Passed
MyPy type checker........................................................Failed
- hook id: mypy
- exit code: 2

modules/orchestration-service/src/__init__.py: error: Duplicate module named
"src" (also at "modules/bot-container/src/__init__.py")
modules/orchestration-service/src/__init__.py: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#mapping-file-paths-to-modules for more info
modules/orchestration-service/src/__init__.py: note: Common resolutions include: a) using `--exclude` to avoid checking one of them, b) adding `__init__.py` somewhere, c) using `--explicit-package-bases` or adjusting MYPYPATH

Bandit security scan.....................................................Failed
- hook id: bandit
- exit code: 1

[main]	INFO	profile include tests: None
[main]	INFO	profile exclude tests: B101,B104,B108
[main]	INFO	cli include tests: None
[main]	INFO	cli exclude tests: None
[main]	INFO	using config: pyproject.toml
[main]	INFO	running on Python 3.12.4
Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04
Run started:2026-01-18 22:27:58.823357

Test results:
>> Issue: [B324:hashlib] Use of weak MD5 hash for security. Consider usedforsecurity=False
   Severity: High   Confidence: High
   CWE: CWE-327 (https://cwe.mitre.org/data/definitions/327.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b324_hashlib.html
   Location: ./modules/orchestration-service/src/audio/config_sync.py:424:15
423	        config_str = json.dumps(config, sort_keys=True)
424	        return hashlib.md5(config_str.encode()).hexdigest()
425	

--------------------------------------------------
>> Issue: [B324:hashlib] Use of weak MD5 hash for security. Consider usedforsecurity=False
   Severity: High   Confidence: High
   CWE: CWE-327 (https://cwe.mitre.org/data/definitions/327.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b324_hashlib.html
   Location: ./modules/orchestration-service/src/audio/translation_cache.py:133:19
132	        content = f"{source_lang}:{target_lang}:{normalized_text}"
133	        hash_val = hashlib.md5(content.encode()).hexdigest()
134	        return f"trans:v1:{hash_val}"

--------------------------------------------------
>> Issue: [B324:hashlib] Use of weak MD5 hash for security. Consider usedforsecurity=False
   Severity: High   Confidence: High
   CWE: CWE-327 (https://cwe.mitre.org/data/definitions/327.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b324_hashlib.html
   Location: ./modules/orchestration-service/src/database/translation_optimization_adapter.py:72:15
71	        content = f"{source_lang}:{target_lang}:{normalized_text}"
72	        return hashlib.md5(content.encode()).hexdigest()
73	

--------------------------------------------------
>> Issue: [B608:hardcoded_sql_expressions] Possible SQL injection vector through string-based query construction.
   Severity: Medium   Confidence: Medium
   CWE: CWE-89 (https://cwe.mitre.org/data/definitions/89.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b608_hardcoded_sql_expressions.html
   Location: ./modules/orchestration-service/src/database/unified_bot_session_repository.py:582:47
581	                for table in tables:
582	                    await db_session.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
583	

--------------------------------------------------
>> Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b614_pytorch_load_save.html
   Location: ./modules/whisper-service/src/eow_detection.py:98:29
97	                map_location = None if torch.cuda.is_available() else torch.device("cpu")
98	                checkpoint = torch.load(cif_ckpt_path, map_location=map_location)
99	                cif_linear.load_state_dict(checkpoint)

--------------------------------------------------
>> Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b614_pytorch_load_save.html
   Location: ./modules/whisper-service/src/silero_vad_iterator.py:225:19
224	    # Load Silero VAD model
225	    model, utils = torch.hub.load(
226	        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
227	    )
228	

--------------------------------------------------
>> Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b614_pytorch_load_save.html
   Location: ./modules/whisper-service/src/simul_whisper/eow_detection.py:22:21
21	            map_location = torch.device("cpu")
22	        checkpoint = torch.load(cfg.cif_ckpt_path, map_location=map_location)
23	        cif_linear.load_state_dict(checkpoint)

--------------------------------------------------
>> Issue: [B310:blacklist] Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected.
   Severity: Medium   Confidence: High
   CWE: CWE-22 (https://cwe.mitre.org/data/definitions/22.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/blacklists/blacklist_calls.html#b310-urllib-urlopen
   Location: ./modules/whisper-service/src/simul_whisper/whisper/__init__.py:75:8
74	    with (
75	        urllib.request.urlopen(url) as source,
76	        open(download_target, "wb") as output,

--------------------------------------------------
>> Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b614_pytorch_load_save.html
   Location: ./modules/whisper-service/src/simul_whisper/whisper/__init__.py:150:21
149	    with io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb") as fp:
150	        checkpoint = torch.load(fp, map_location=device)
151	    del checkpoint_file

--------------------------------------------------
>> Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b614_pytorch_load_save.html
   Location: ./modules/whisper-service/src/vac_online_processor.py:188:27
187	        try:
188	            vad_model, _ = torch.hub.load(
189	                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
190	            )
191	            self.vad = FixedVADIterator(

--------------------------------------------------
>> Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/0.0.0/plugins/b614_pytorch_load_save.html
   Location: ./modules/whisper-service/src/vad_detector.py:228:23
227	
228	            model, _ = torch.hub.load(
229	                repo_or_dir="snakers4/silero-vad",
230	                model="silero_vad",
231	                force_reload=False,
232	                onnx=False,
233	            )
234	            model.eval()

--------------------------------------------------

Code scanned:
	Total lines of code: 91457
	Total lines skipped (#nosec): 0
	Total potential issues skipped due to specifically being disabled (e.g., #nosec BXXX): 0

Run metrics:
	Total issues (by severity):
		Undefined: 0
		Low: 25
		Medium: 11
		High: 3
	Total issues (by confidence):
		Undefined: 0
		Low: 3
		Medium: 4
		High: 32
Files skipped (0):

Detect secrets...........................................................Failed
- hook id: detect-secrets
- exit code: 1

ERROR: Potential secrets about to be committed to git repo!

Secret Type: Basic Auth Credentials
Location:    docs/archive/planning/UPGRADE_PLAN.md:139

Secret Type: Secret Keyword
Location:    meeting-bot/README.md:203

Secret Type: Secret Keyword
Location:    modules/bot-container/tests/integration/test_simple_join.py:26

Secret Type: Hex High Entropy String
Location:    modules/orchestration-service/alembic/versions/002_make_session_events_session_id_nullable.py:22

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/e2e/test_multiple_meeting_codes.py:46

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/integration/test_pipeline_dry_integration.py:41

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_fireflies_router.py:103

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/tests/integration/test_audio_coordinator_optimization.py:33

Secret Type: Hex High Entropy String
Location:    reference/vexa/docs/self-hosted-management.md:87

Secret Type: Hex High Entropy String
Location:    reference/vexa/libs/shared-models/alembic/versions/dc59a1c03d1f_add_meeting_data_jsonb_column.py:13

Secret Type: Secret Keyword
Location:    reference/vexa/services/mcp/README.md:53

Secret Type: Basic Auth Credentials
Location:    tests/integration/conftest.py:28

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Basic Auth Credentials
Location:    README-DATABASE.md:94

Secret Type: Basic Auth Credentials
Location:    README-DATABASE.md:191

Secret Type: Secret Keyword
Location:    modules/orchestration-service/docs/fixes/FIXES_APPLIED.md:13

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/src/audio/database_adapter.py:802

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_fireflies_models.py:247

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_fireflies_models.py:264

Secret Type: Basic Auth Credentials
Location:    modules/whisper-service/ffmpeg/doc/ffplay-all.html:10429

Secret Type: Secret Keyword
Location:    reference/vexa/docs/user_api_guide.md:54

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Secret Keyword
Location:    docs/audit_findings/orchestration_service_security.md:68

Secret Type: Secret Keyword
Location:    docs/audit_findings/orchestration_service_security.md:567

Secret Type: Secret Keyword
Location:    modules/orchestration-service/IMPLEMENTATION_SUMMARY.md:301

Secret Type: Secret Keyword
Location:    modules/orchestration-service/PRODUCTION_FIXES_SUMMARY.md:436

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/tests/audio/README.md:105

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/mocks/fireflies_mock_server.py:18

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/tests/integration/test_translation_persistence.py:28

Secret Type: Base64 High Entropy String
Location:    modules/whisper-service/ffmpeg/doc/faq.html:52

Secret Type: Base64 High Entropy String
Location:    modules/whisper-service/ffmpeg/doc/faq.html:53

Secret Type: Base64 High Entropy String
Location:    modules/whisper-service/ffmpeg/doc/faq.html:82

Secret Type: Base64 High Entropy String
Location:    modules/whisper-service/ffmpeg/doc/faq.html:365

Secret Type: Base64 High Entropy String
Location:    modules/whisper-service/ffmpeg/doc/faq.html:371

Secret Type: Base64 High Entropy String
Location:    modules/whisper-service/ffmpeg/doc/faq.html:768

Secret Type: Secret Keyword
Location:    tests/integration/docker-compose.test.yml:14

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Secret Keyword
Location:    modules/meeting-bot-service/README.md:203

Secret Type: Hex High Entropy String
Location:    modules/orchestration-service/alembic/versions/5f3bcf8a26da_add_all_missing_tables_and_indexes.py:14

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/scripts/setup_database.sh:26

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/integration/test_mock_server_api_contract.py:553

Secret Type: Basic Auth Credentials
Location:    modules/whisper-service/ffmpeg/doc/ffmpeg-protocols.html:1233

Secret Type: Basic Auth Credentials
Location:    modules/whisper-service/ffmpeg/doc/ffprobe-all.html:10782

Secret Type: Basic Auth Credentials
Location:    reference/vexa/services/WhisperLive/README.md:130

Secret Type: Secret Keyword
Location:    scripts/setup_postgres.sh:17

Secret Type: Secret Keyword
Location:    scripts/start-database.sh:253

Secret Type: Secret Keyword
Location:    scripts/start-database.sh:254

Secret Type: Secret Keyword
Location:    scripts/start-database.sh:256

Secret Type: Secret Keyword
Location:    scripts/start-database.sh:257

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Secret Keyword
Location:    modules/orchestration-service/DATA_PIPELINE_README.md:267

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/src/audio/chunk_manager.py:750

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/e2e/test_full_pipeline_e2e.py:53

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_sentence_aggregator.py:75

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_sentence_aggregator.py:140

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/tests/integration/test_audio_orchestration.py:51

Secret Type: Base64 High Entropy String
Location:    reference/vexa/services/mcp/config.json:13

Secret Type: Secret Keyword
Location:    reference/vexa/services/mcp/config.json:13

Secret Type: Secret Keyword
Location:    reference/vexa/testing/run_vexa_interaction.sh:194

Secret Type: Secret Keyword
Location:    scripts/quick_db_setup.sh:30

Secret Type: Secret Keyword
Location:    scripts/quick_db_setup.sh:48

Secret Type: Basic Auth Credentials
Location:    tests/integration/README.md:224

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Basic Auth Credentials
Location:    .env.database:22

Secret Type: Basic Auth Credentials
Location:    docs/archive/planning/COORDINATION_EXECUTION_SUMMARY.md:23

Secret Type: Secret Keyword
Location:    modules/orchestration-service/PIPELINE_INTEGRATION_SUMMARY.md:197

Secret Type: Secret Keyword
Location:    modules/orchestration-service/src/bot/bot_manager.py:323

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/src/managers/config_manager.py:336

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/integration/test_fireflies_dashboard_api.py:334

Secret Type: Basic Auth Credentials
Location:    modules/whisper-service/README.md:172

Secret Type: Secret Keyword
Location:    modules/whisper-service/README.md:524

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Secret Keyword
Location:    docker-compose.database.yml:73

Secret Type: Secret Keyword
Location:    docs/archive/analysis/DATABASE_OPTIMIZATION_SETUP.md:495

Secret Type: Base64 High Entropy String
Location:    modules/orchestration-service/docs/reference/GOOGLE_MEET_REFERENCES.md:950

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/scripts/setup_database.py:10

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/src/audio/audio_coordinator.py:2343

Secret Type: Secret Keyword
Location:    modules/orchestration-service/src/pipeline/data_pipeline.py:952

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/conftest.py:289

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/integration/test_fireflies_integration.py:86

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/integration/test_fireflies_integration.py:279

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/integration/test_fireflies_integration.py:283

Secret Type: Base64 High Entropy String
Location:    modules/whisper-service/ffmpeg/doc/platform.html:267

Secret Type: Hex High Entropy String
Location:    reference/vexa/libs/shared-models/alembic/versions/5befe308fa8b_add_data_field_to_users_table.py:15

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Secret Keyword
Location:    env.template:42

Secret Type: Secret Keyword
Location:    env.template:45

Secret Type: Secret Keyword
Location:    env.template:49

Secret Type: Secret Keyword
Location:    env.template:52

Secret Type: Secret Keyword
Location:    env.template:118

Secret Type: Secret Keyword
Location:    env.template:123

Secret Type: Secret Keyword
Location:    env.template:156

Secret Type: Secret Keyword
Location:    env.template:194

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/docs/configuration/CONFIGURATION_GUIDE.md:263

Secret Type: Secret Keyword
Location:    modules/orchestration-service/src/models/fireflies.py:234

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/tests/conftest.py:23

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_obs_output.py:54

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_obs_output.py:93

Secret Type: Basic Auth Credentials
Location:    modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:23945

Secret Type: Hex High Entropy String
Location:    reference/vexa/Makefile:354

Secret Type: Secret Keyword
Location:    reference/vexa/docs/websocket.md:343

Secret Type: Secret Keyword
Location:    reference/vexa/services/transcription-collector/config.py:27

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Secret Keyword
Location:    docs/archive/planning/QUICK_START_DATABASE.md:23

Secret Type: Secret Keyword
Location:    docs/archive/planning/QUICK_START_DATABASE.md:36

Secret Type: Secret Keyword
Location:    modules/bot-container/AUTHENTICATION.md:76

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/alembic.ini:61

Secret Type: Secret Keyword
Location:    modules/orchestration-service/monitoring/alertmanager/alertmanager.yml:5

Secret Type: Secret Keyword
Location:    modules/orchestration-service/monitoring/alertmanager/alertmanager.yml:48

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:45

Secret Type: Hex High Entropy String
Location:    reference/vexa/services/WhisperLive/docs/html/.buildinfo:3

Secret Type: Hex High Entropy String
Location:    reference/vexa/services/WhisperLive/docs/html/.buildinfo:4

Secret Type: Hex High Entropy String
Location:    reference/vexa/services/WhisperLive/docs/html/searchindex.js:1

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository
ERROR: Potential secrets about to be committed to git repo!

Secret Type: Basic Auth Credentials
Location:    compose.local.yml:14

Secret Type: Secret Keyword
Location:    compose.local.yml:167

Secret Type: Basic Auth Credentials
Location:    docs/guides/database-setup.md:147

Secret Type: Secret Keyword
Location:    docs/guides/database-setup.md:288

Secret Type: Basic Auth Credentials
Location:    docs/guides/database-setup.md:395

Secret Type: Basic Auth Credentials
Location:    modules/orchestration-service/PIPELINE_INTEGRATION_COMPLETE.md:650

Secret Type: Secret Keyword
Location:    modules/orchestration-service/src/database/bot_session_manager.py:1117

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/e2e/test_import_pipeline_e2e.py:284

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/e2e/test_import_pipeline_e2e.py:361

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/integration/test_glossary_integration.py:302

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_fireflies_client.py:67

Secret Type: Secret Keyword
Location:    modules/orchestration-service/tests/fireflies/unit/test_fireflies_client.py:101

Secret Type: Secret Keyword
Location:    modules/translation-service/CLAUDE.md:515

Secret Type: Secret Keyword
Location:    modules/translation-service/CLAUDE.md:518

Secret Type: Secret Keyword
Location:    modules/translation-service/README.md:887

Secret Type: Secret Keyword
Location:    modules/translation-service/README.md:890

Secret Type: Secret Keyword
Location:    modules/translation-service/src/openai_compatible_translator.py:723

Secret Type: Secret Keyword
Location:    modules/translation-service/src/openai_compatible_translator.py:729

Secret Type: Secret Keyword
Location:    modules/translation-service/src/openai_compatible_translator.py:775

Possible mitigations:
  - For information about putting your secrets in a safer place, please ask in
    #security
  - Mark false positives with an inline `pragma: allowlist secret` comment

If a secret has already been committed, visit
https://help.github.com/articles/removing-sensitive-data-from-a-repository

Gitleaks secret scan.....................................................Passed
Check spelling...........................................................Failed
- hook id: codespell
- exit code: 65

meeting-bot/CODE_OF_CONDUCT.md:8: socio-economic ==> socioeconomic
meeting-bot/package-lock.json:4804: bU ==> by, be, but, bug, bun, bud, buy, bum
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:39: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:101: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:137: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:210: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:247: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:253: fases ==> fazes, phases
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:259: fase ==> faze, phase, false
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:271: fase ==> faze, phase, false
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:283: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:320: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:338: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:374: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:393: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/fixtures/meeting_transcript_5min.py:463: cliente ==> client, clientele
modules/whisper-service/IMPLEMENTATION_PLAN.md:139: FEEBACK ==> FEEDBACK
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:2101: Japanse ==> Japanese
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:2160: internaly ==> internally
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:2180: charactor ==> character
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:3635: optimzations ==> optimizations
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:4552: paeth ==> path
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:6584: paeth ==> path
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:6584: paeth ==> path
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:6778: durning ==> during
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:6908: feautre ==> feature
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:7154: feautre ==> feature
modules/whisper-service/ffmpeg/doc/ffmpeg-codecs.html:7440: feautre ==> feature
modules/whisper-service/ffmpeg/doc/ffmpeg-protocols.html:181: Muliple ==> Multiple
modules/whisper-service/ffmpeg/doc/ffmpeg-protocols.html:788: prefered ==> preferred
modules/whisper-service/ffmpeg/doc/ffmpeg-protocols.html:2084: wasn ==> wasn't, was
modules/whisper-service/ffmpeg/doc/ffmpeg.html:33: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg.html:33: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg.html:415: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg.html:416: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg.html:416: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg.html:481: trancoding ==> transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg.html:1611: Wether ==> Weather, Whether
modules/whisper-service/ffmpeg/doc/ffmpeg.html:2114: setts ==> sets
modules/whisper-service/tests/integration/test_domain_prompt_integration.py:54: infarction ==> infraction
modules/whisper-service/tests/integration/test_domain_prompt_integration.py:92: infarction ==> infraction
modules/whisper-service/tests/integration/test_domain_prompt_integration.py:322: infarction ==> infraction
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:1392: IHs ==> his
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:2015: Lik ==> Like, Lick, Link
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:3866: OTU ==> OUT
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:4522: ODY ==> BODY
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:14991: VEw ==> view, vow, vex
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:24329: wqs ==> was
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:27866: eGE ==> edge
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:29165: Ons ==> Owns
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:32458: WEw ==> we
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:33187: SmE ==> same, seme, some, sms
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:40493: UEs ==> yes, use
reference/SimulStreaming/simul_whisper/whisper/assets/gpt2.tiktoken:49579: THQ ==> THE
modules/meeting-bot-service/package-lock.json:4804: bU ==> by, be, but, bug, bun, bud, buy, bum
modules/whisper-service/ffmpeg/doc/ffmpeg-devices.html:960: imput ==> input
modules/whisper-service/ffmpeg/doc/general.html:258: enhacement ==> enhancement
modules/whisper-service/ffmpeg/doc/general.html:609: LAF ==> LAUGH, LEAF, LOAF, LAD, LAG, LAC, KAF, KAPH
modules/whisper-service/ffmpeg/doc/general.html:746: SER ==> SET
modules/whisper-service/ffmpeg/doc/general.html:1192: ALS ==> ALSO
modules/whisper-service/src/simul_whisper/simul_whisper.py:129: suppresed ==> suppressed
modules/whisper-service/src/simul_whisper/simul_whisper.py:552: supress ==> suppress
modules/whisper-service/src/simul_whisper/simul_whisper.py:650: ommit ==> omit
modules/whisper-service/src/simul_whisper/whisper/tokenizer.py:50: te ==> the, be, we, to
modules/whisper-service/src/simul_whisper/whisper/tokenizer.py:89: fo ==> of, for, to, do, go
modules/whisper-service/src/simul_whisper/whisper/trans_nopad.py:291: sentenses ==> sentences
modules/whisper-service/src/simul_whisper/whisper/trans_nopad.py:437: supercedes ==> supersedes
reference/SimulStreaming/simul_whisper/whisper/normalizers/english.py:61: nd ==> and, 2nd
reference/SimulStreaming/simul_whisper/whisper/normalizers/english.py:413: nd ==> and, 2nd
reference/SimulStreaming/simul_whisper/whisper/trans_nopad.py:292: sentenses ==> sentences
reference/SimulStreaming/simul_whisper/whisper/trans_nopad.py:455: supercedes ==> supersedes
reference/vexa/libs/shared-models/shared_models/schemas.py:16: fo ==> of, for, to, do, go
reference/vexa/libs/shared-models/shared_models/schemas.py:20: te ==> the, be, we, to
reference/vexa/services/WhisperLive/docs/html/_static/pygments.css:25: ges ==> goes, guess
reference/vexa/services/WhisperLive/docs/html/_static/pygments.css:47: nd ==> and, 2nd
modules/meeting-bot-service/src/bots/ZoomBot.ts:445: annoucements ==> announcements
modules/orchestration-service/src/system_constants.py:67: te ==> the, be, we, to
modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py:182: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py:273: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py:274: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py:304: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py:755: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_speaker_correlator.py:756: assertIn ==> asserting, assert in, assertion
modules/whisper-service/ffmpeg/doc/ffmpeg-bitstream-filters.html:62: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-bitstream-filters.html:62: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-bitstream-filters.html:1071: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-bitstream-filters.html:1072: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-bitstream-filters.html:1072: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-bitstream-filters.html:1183: setts ==> sets
reference/SimulStreaming/simul_whisper/simul_whisper.py:116: suppresed ==> suppressed
reference/SimulStreaming/simul_whisper/simul_whisper.py:440: supress ==> suppress
reference/SimulStreaming/simul_whisper/simul_whisper.py:515: ommit ==> omit
reference/SimulStreaming/translate/sentence_segmenter.py:23: preceeded ==> preceded, proceeded
modules/meeting-bot-service/CODE_OF_CONDUCT.md:8: socio-economic ==> socioeconomic
modules/whisper-service/docs/legacy/CODE_SWITCHING_ANALYSIS.md:436: bu ==> by, be, but, bug, bun, bud, buy, bum
modules/whisper-service/docs/legacy/buffer_plan.md:823: toi ==> to, toy
modules/whisper-service/src/simul_whisper/eow_detection.py:39: threashold ==> threshold
modules/whisper-service/src/simul_whisper/eow_detection.py:67: intergrate ==> integrate
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:1392: IHs ==> his
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:2015: Lik ==> Like, Lick, Link
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:3866: OTU ==> OUT
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:4522: ODY ==> BODY
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:14991: VEw ==> view, vow, vex
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:24329: wqs ==> was
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:27866: eGE ==> edge
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:29165: Ons ==> Owns
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:32458: WEw ==> we
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:33187: SmE ==> same, seme, some, sms
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:40493: UEs ==> yes, use
modules/whisper-service/src/simul_whisper/whisper/assets/gpt2.tiktoken:49579: THQ ==> THE
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:4274: Lik ==> Like, Lick, Link
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:10930: IHs ==> his
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:15414: INeS ==> lines
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:15719: OTU ==> OUT
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:19112: SmE ==> same, seme, some, sms
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:22194: ODY ==> BODY
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:36785: wqs ==> was
modules/whisper-service/src/simul_whisper/whisper/assets/multilingual.tiktoken:36853: eGE ==> edge
reference/SimulStreaming/README.md:169: miliseconds ==> milliseconds
reference/vexa/services/WhisperLive/Audio-Transcription-Chrome/README.md:36: recieved ==> received
reference/vexa/services/WhisperLive/docs/html/_static/language_data.js:49: ative ==> active, native
reference/vexa/services/WhisperLive/docs/html/_static/language_data.js:147: ative ==> active, native
reference/vexa/services/WhisperLive/docs/html/_static/language_data.js:158: ment ==> meant
reference/vexa/services/vexa-bot/package-lock.json:1153: bU ==> by, be, but, bug, bun, bud, buy, bum
CODE_OF_CONDUCT.md:8: socio-economic ==> socioeconomic
modules/orchestration-service/src/database/database.py:84: checkin ==> checking, check in
modules/orchestration-service/src/database/database.py:86: checkin ==> checking, check in
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:286: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:287: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:312: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:387: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:485: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:487: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:491: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:492: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:493: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:494: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:515: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:609: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:610: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:611: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:612: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:693: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:694: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:698: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/audio/unit/test_timing_coordinator.py:702: assertIn ==> asserting, assert in, assertion
modules/orchestration-service/tests/e2e/test_caption_pipeline_e2e.py:117: fase ==> faze, phase, false
modules/orchestration-service/tests/e2e/test_caption_pipeline_e2e.py:118: fase ==> faze, phase, false
modules/orchestration-service/tests/e2e/test_caption_pipeline_e2e.py:119: fase ==> faze, phase, false
modules/whisper-service/src/simul_whisper/beam.py:4: extention ==> extension
reference/SimulStreaming/simulstreaming_whisper.py:191: imcomplete ==> incomplete
reference/vexa/services/WhisperLive/whisper_live/server.py:1781: hasnt ==> hasn't
reference/vexa/services/WhisperLive/whisper_live/server.py:2276: reptition ==> repetition
reference/vexa/services/WhisperLive/whisper_live/server.py:2277: thats ==> that's
reference/vexa/services/WhisperLive/whisper_live/server.py:2738: reptition ==> repetition
reference/vexa/services/WhisperLive/whisper_live/server.py:2739: thats ==> that's
modules/seamless/seamless_readme.md:167: foward ==> forward
modules/seamless/seamless_readme.md:220: calcuate ==> calculate
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:162: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:162: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:1662: HiLight ==> highlight
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:1782: Re-use ==> Reuse
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:2352: specied ==> specified
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:2890: infinte ==> infinite
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:3116: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:3117: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:3117: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:3118: CAF ==> CALF
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:4013: re-use ==> reuse
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:4104: throuhg ==> through
modules/whisper-service/ffmpeg/doc/ffmpeg-formats.html:4838: desgined ==> designed
reference/SimulStreaming/simul_whisper/beam.py:3: extention ==> extension
reference/vexa/services/WhisperLive/Audio-Transcription-Firefox/README.md:34: recieved ==> received
modules/whisper-service/ffmpeg/README.txt:163: als ==> also
modules/whisper-service/ffmpeg/README.txt:169: anull ==> annul
modules/whisper-service/ffmpeg/README.txt:308: anull ==> annul
modules/whisper-service/ffmpeg/README.txt:462: ser ==> set
modules/whisper-service/ffmpeg/README.txt:467: caf ==> calf
modules/whisper-service/ffmpeg/README.txt:469: laf ==> laugh, leaf, loaf, lad, lag, lac, kaf, kaph
modules/whisper-service/ffmpeg/README.txt:560: caf ==> calf
modules/whisper-service/ffmpeg/README.txt:674: anull ==> annul
modules/whisper-service/ffmpeg/README.txt:804: setts ==> sets
reference/SimulStreaming/simul_whisper/eow_detection.py:38: threashold ==> threshold
reference/SimulStreaming/simul_whisper/eow_detection.py:63: intergrate ==> integrate
reference/SimulStreaming/simul_whisper/whisper/tokenizer.py:51: te ==> the, be, we, to
reference/SimulStreaming/simul_whisper/whisper/tokenizer.py:90: fo ==> of, for, to, do, go
reference/SimulStreaming/simul_whisper/whisper/transcribe.py:421: supercedes ==> supersedes
reference/vexa/services/WhisperLive/Audio-Transcription-Chrome/popup.html:48: fo ==> of, for, to, do, go
reference/vexa/services/WhisperLive/Audio-Transcription-Chrome/popup.html:113: te ==> the, be, we, to
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: devic ==> device
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: fals ==> fails, falls, false
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: languag ==> language
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: attribut ==> attribute
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: messag ==> message
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: uniqu ==> unique
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: previou ==> previous
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: includ ==> include
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: thi ==> the, this
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: respons ==> response, respond
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: thei ==> their, they
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: receiv ==> receive
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: specifi ==> specific, specify
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: usag ==> usage
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: provid ==> provide, prove, proved, proves
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: befor ==> before
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: destroi ==> destroy
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: combin ==> combing, comb in, combine
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: complet ==> complete
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: preced ==> precede
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: singl ==> single
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: infinit ==> infinite
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: histori ==> history, historic
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: commun ==> commune, common
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: assum ==> assume
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: updat ==> update
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: multipl ==> multiple, multiply
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: modul ==> module
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: activ ==> active
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: determin ==> determine
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: requir ==> require
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: valu ==> value
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: delet ==> delete
reference/vexa/services/WhisperLive/docs/html/searchindex.js:1: tabl ==> table
reference/vexa/services/vexa-bot/core/src/constans.ts: constans ==> constants, constant, constance
tests/integration/test_cif_word_boundaries.py:130: te ==> the, be, we, to
tests/integration/test_cif_word_boundaries.py:135: te ==> the, be, we, to
meeting-bot/src/bots/ZoomBot.ts:445: annoucements ==> announcements
modules/translation-service/CLAUDE.md:32: Fallbck ==> Fallback
modules/whisper-service/docs/legacy/PLAN.md:180: ANE ==> AND
modules/whisper-service/docs/legacy/PLAN.md:211: ANE ==> AND
modules/whisper-service/docs/legacy/PLAN.md:347: ANE ==> AND
modules/whisper-service/docs/legacy/PLAN.md:657: ANE ==> AND
modules/whisper-service/ffmpeg/doc/ffplay-all.html:174: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffplay-all.html:174: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffplay-all.html:540: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffplay-all.html:540: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffplay-all.html:1163: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:1164: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:6094: Japanse ==> Japanese
modules/whisper-service/ffmpeg/doc/ffplay-all.html:6153: internaly ==> internally
modules/whisper-service/ffmpeg/doc/ffplay-all.html:6173: charactor ==> character
modules/whisper-service/ffmpeg/doc/ffplay-all.html:7393: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffplay-all.html:7394: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffplay-all.html:7394: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffplay-all.html:7505: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffplay-all.html:8921: HiLight ==> highlight
modules/whisper-service/ffmpeg/doc/ffplay-all.html:9041: Re-use ==> Reuse
modules/whisper-service/ffmpeg/doc/ffplay-all.html:9377: Muliple ==> Multiple
modules/whisper-service/ffmpeg/doc/ffplay-all.html:9984: prefered ==> preferred
modules/whisper-service/ffmpeg/doc/ffplay-all.html:11280: wasn ==> wasn't, was
modules/whisper-service/ffmpeg/doc/ffplay-all.html:12567: imput ==> input
modules/whisper-service/ffmpeg/doc/ffplay-all.html:14205: redability ==> readability
modules/whisper-service/ffmpeg/doc/ffplay-all.html:15845: fo ==> of, for, to, do, go
modules/whisper-service/ffmpeg/doc/ffplay-all.html:17184: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffplay-all.html:17185: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffplay-all.html:17185: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffplay-all.html:17889: trough ==> through
modules/whisper-service/ffmpeg/doc/ffplay-all.html:18079: trough ==> through
modules/whisper-service/ffmpeg/doc/ffplay-all.html:23125: custon ==> custom
modules/whisper-service/ffmpeg/doc/ffplay-all.html:24580: commerical ==> commercial
modules/whisper-service/ffmpeg/doc/ffplay-all.html:24743: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffplay-all.html:24750: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffplay-all.html:24757: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31134: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31136: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31137: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31162: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31163: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31174: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31176: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31188: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31193: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31195: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31196: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31199: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31201: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:31216: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:32697: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffplay-all.html:34895: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffplay-all.html:34933: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffplay-all.html:36590: overlayed ==> overlaid
modules/whisper-service/ffmpeg/doc/ffplay-all.html:40343: backgroud ==> background
modules/whisper-service/ffmpeg/doc/ffplay-all.html:40375: preseved ==> preserved
modules/whisper-service/ffmpeg/doc/ffplay-all.html:43057: streched ==> stretched
modules/whisper-service/ffmpeg/doc/ffplay-all.html:43689: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffplay-all.html:43903: rin ==> ring, rink, rind, rain, rein, ruin, grin
modules/whisper-service/ffmpeg/doc/ffplay-all.html:48025: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:48063: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffplay-all.html:48626: acount ==> account
modules/whisper-service/ffmpeg/doc/ffplay-all.html:49887: TE ==> THE, BE, WE, TO
modules/whisper-service/ffmpeg/doc/ffplay-all.html:49892: TE ==> THE, BE, WE, TO
modules/whisper-service/ffmpeg/doc/ffplay-all.html:52308: enhacement ==> enhancement
modules/whisper-service/ffmpeg/doc/ffplay-all.html:52659: LAF ==> LAUGH, LEAF, LOAF, LAD, LAG, LAC, KAF, KAPH
modules/whisper-service/ffmpeg/doc/ffplay-all.html:52796: SER ==> SET
modules/whisper-service/ffmpeg/doc/ffplay-all.html:53242: ALS ==> ALSO
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:182: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:182: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:548: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:548: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:1171: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:1172: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:6447: Japanse ==> Japanese
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:6506: internaly ==> internally
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:6526: charactor ==> character
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:7746: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:7747: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:7747: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:7858: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:9274: HiLight ==> highlight
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:9394: Re-use ==> Reuse
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:9730: Muliple ==> Multiple
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:10337: prefered ==> preferred
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:11633: wasn ==> wasn't, was
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:12920: imput ==> input
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:14558: redability ==> readability
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:16198: fo ==> of, for, to, do, go
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:17537: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:17538: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:17538: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:18242: trough ==> through
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:18432: trough ==> through
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:23478: custon ==> custom
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:24933: commerical ==> commercial
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:25096: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:25103: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:25110: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31487: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31489: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31490: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31515: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31516: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31527: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31529: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31541: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31546: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31548: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31549: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31552: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31554: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:31569: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:33050: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:35248: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:35286: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:36943: overlayed ==> overlaid
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:40696: backgroud ==> background
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:40728: preseved ==> preserved
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:43410: streched ==> stretched
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:44042: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:44256: rin ==> ring, rink, rind, rain, rein, ruin, grin
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:48378: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:48416: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:48979: acount ==> account
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:50240: TE ==> THE, BE, WE, TO
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:50245: TE ==> THE, BE, WE, TO
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:52661: enhacement ==> enhancement
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:53012: LAF ==> LAUGH, LEAF, LOAF, LAD, LAG, LAC, KAF, KAPH
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:53149: SER ==> SET
modules/whisper-service/ffmpeg/doc/ffprobe-all.html:53595: ALS ==> ALSO
modules/whisper-service/legacy/simul_whisper_WORKING.py:129: suppresed ==> suppressed
modules/whisper-service/legacy/simul_whisper_WORKING.py:553: supress ==> suppress
modules/whisper-service/legacy/simul_whisper_WORKING.py:639: ommit ==> omit
modules/funasr/funasr_readme.md:41: SER ==> SET
modules/funasr/funasr_readme.md:105: SER ==> SET
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:117: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:153: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:263: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:269: fases ==> fazes, phases
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:275: fase ==> faze, phase, false
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:287: fase ==> faze, phase, false
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:354: cliente ==> client, clientele
modules/orchestration-service/tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py:449: cliente ==> client, clientele
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:33: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:33: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:439: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:439: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:582: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:582: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:1069: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:1069: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:1692: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:1693: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:2912: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:2913: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:2913: Trancoding ==> Transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:2978: trancoding ==> transcoding
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:4108: Wether ==> Weather, Whether
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:4611: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:9759: Japanse ==> Japanese
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:9818: internaly ==> internally
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:9838: charactor ==> character
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:11293: optimzations ==> optimizations
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:12210: paeth ==> path
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:14242: paeth ==> path
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:14242: paeth ==> path
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:14436: durning ==> during
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:14566: feautre ==> feature
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:14812: feautre ==> feature
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:15098: feautre ==> feature
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:16505: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:16506: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:16506: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:16617: setts ==> sets
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:18033: HiLight ==> highlight
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:18153: Re-use ==> Reuse
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:18723: specied ==> specified
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:19261: infinte ==> infinite
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:19487: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:19488: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:19488: caf ==> calf
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:19489: CAF ==> CALF
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:20384: re-use ==> reuse
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:20475: throuhg ==> through
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:21209: desgined ==> designed
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:22893: Muliple ==> Multiple
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:23500: prefered ==> preferred
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:24796: wasn ==> wasn't, was
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:26083: imput ==> input
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:28205: redability ==> readability
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:29845: fo ==> of, for, to, do, go
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:31184: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:31185: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:31185: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:31889: trough ==> through
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:32079: trough ==> through
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:37125: custon ==> custom
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:38580: commerical ==> commercial
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:38743: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:38750: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:38757: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45134: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45136: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45137: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45162: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45163: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45174: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45176: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45188: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45193: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45195: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45196: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45199: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45201: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:45216: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:46697: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:48895: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:48933: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:50590: overlayed ==> overlaid
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:54343: backgroud ==> background
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:54375: preseved ==> preserved
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:57057: streched ==> stretched
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:57689: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:57903: rin ==> ring, rink, rind, rain, rein, ruin, grin
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:62025: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:62063: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:62626: acount ==> account
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:63887: TE ==> THE, BE, WE, TO
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:63892: TE ==> THE, BE, WE, TO
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:66308: enhacement ==> enhancement
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:66659: LAF ==> LAUGH, LEAF, LOAF, LAD, LAG, LAC, KAF, KAPH
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:66796: SER ==> SET
modules/whisper-service/ffmpeg/doc/ffmpeg-all.html:67242: ALS ==> ALSO
modules/whisper-service/src/session_restart/session_manager.py:47: FEEBACK ==> FEEDBACK
modules/whisper-service/src/session_restart/session_manager.py:182: FEEBACK ==> FEEDBACK
modules/whisper-service/src/session_restart/session_manager.py:305: FEEBACK ==> FEEDBACK
modules/whisper-service/src/simul_whisper/whisper/normalizers/english.py:62: nd ==> and, 2nd
modules/whisper-service/src/simul_whisper/whisper/normalizers/english.py:408: nd ==> and, 2nd
modules/whisper-service/tests/unit/test_token_buffer.py:404: infarction ==> infraction
reference/SimulStreaming/translate/simul_llm_translate.py:347: Sie ==> Size, Sigh, Side
reference/vexa/services/vexa-bot/core/package-lock.json:1273: bU ==> by, be, but, bug, bun, bud, buy, bum
reference/vexa/services/vexa-bot/core/src/index.ts:7: constans ==> constants, constant, constance
modules/orchestration-service/docs/reference/GOOGLE_MEET_REFERENCES.md:648: re-use ==> reuse
modules/whisper-service/ffmpeg/doc/community.html:166: recuse ==> recurse
modules/whisper-service/ffmpeg/doc/developer.html:604: outweight ==> outweigh
modules/whisper-service/ffmpeg/doc/developer.html:654: preprocesor ==> preprocessor
modules/whisper-service/ffmpeg/doc/developer.html:936: theres ==> there's
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:167: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:167: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:790: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:791: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:1845: redability ==> readability
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:3485: fo ==> of, for, to, do, go
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:4824: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:4825: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:4825: anull ==> annul
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:5529: trough ==> through
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:5719: trough ==> through
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:10765: custon ==> custom
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:12220: commerical ==> commercial
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:12383: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:12390: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:12397: neigbour ==> neighbour
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18774: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18776: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18777: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18802: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18803: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18814: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18816: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18828: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18833: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18835: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18836: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18839: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18841: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:18856: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:20337: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:22535: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:22573: alls ==> all, falls
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:24230: overlayed ==> overlaid
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:27983: backgroud ==> background
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:28015: preseved ==> preserved
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:30697: streched ==> stretched
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:31329: ot ==> to, of, or, not, it
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:31543: rin ==> ring, rink, rind, rain, rein, ruin, grin
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:35665: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:35703: Hald ==> Held, Hold, Half, Hall
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:36266: acount ==> account
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:37527: TE ==> THE, BE, WE, TO
modules/whisper-service/ffmpeg/doc/ffmpeg-filters.html:37532: TE ==> THE, BE, WE, TO
modules/whisper-service/src/simul_whisper/whisper/transcribe.py:388: supercedes ==> supersedes
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:4274: Lik ==> Like, Lick, Link
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:10930: IHs ==> his
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:15414: INeS ==> lines
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:15719: OTU ==> OUT
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:19112: SmE ==> same, seme, some, sms
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:22194: ODY ==> BODY
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:36785: wqs ==> was
reference/SimulStreaming/simul_whisper/whisper/assets/multilingual.tiktoken:36853: eGE ==> edge
reference/SimulStreaming/whisper_streaming/base.py:5: neeeded ==> needed
reference/vexa/services/WhisperLive/Audio-Transcription-Firefox/popup.html:49: fo ==> of, for, to, do, go
reference/vexa/services/WhisperLive/Audio-Transcription-Firefox/popup.html:114: te ==> the, be, we, to
scripts/deploy.py:64: renderD ==> rendered
scripts/deploy.py:110: fo ==> of, for, to, do, go

ShellCheck...............................................................Failed
- hook id: shellcheck
- exit code: 1

In meeting-bot/scripts/test-production-build.sh line 17:
CONTAINER_ID=$(docker run -d --name meeting-bot-test -p 3001:3000 meeting-bot:test)
^----------^ SC2034 (warning): CONTAINER_ID appears unused. Verify use (or export if used externally).


In modules/meeting-bot-service/scripts/test-production-build.sh line 17:
CONTAINER_ID=$(docker run -d --name meeting-bot-test -p 3001:3000 meeting-bot:test)
^----------^ SC2034 (warning): CONTAINER_ID appears unused. Verify use (or export if used externally).


In modules/translation-service/start-translation-service.sh line 22:
BLUE='\033[0;34m'
^--^ SC2034 (warning): BLUE appears unused. Verify use (or export if used externally).


In modules/translation-service/start-translation-service.sh line 109:
    PIP_CMD="pip"
    ^-----^ SC2034 (warning): PIP_CMD appears unused. Verify use (or export if used externally).


In reference/vexa/services/WhisperLive/scripts/build_whisper_tensorrt.sh line 75:
        $( [[ "$weight_only_precision" == "int8" || "$weight_only_precision" == "int4" ]] && echo "--use_weight_only --weight_only_precision $weight_only_precision" ) \
        ^-- SC2046 (warning): Quote this to prevent word splitting.


In reference/vexa/services/WhisperLive/scripts/build_whisper_tensorrt.sh line 119:
cd $tensorrt_examples_dir/whisper
^-- SC2164 (warning): Use 'cd ... || exit' or 'cd ... || return' in case cd fails.

Did you mean: 
cd $tensorrt_examples_dir/whisper || exit


In reference/vexa/services/vexa-bot/core/src/platforms/hot-debug.sh line 175:
cleanup_and_exit 0
^----------------^ SC2218 (error): This function is only defined later. Move the definition up.


In scripts/check_docker_env.sh line 209:
    local disk_available
          ^------------^ SC2034 (warning): disk_available appears unused. Verify use (or export if used externally).


In scripts/load-secrets.sh line 6:
    export JWT_SECRET=$(cat .secrets/jwt_secret.txt)
           ^--------^ SC2155 (warning): Declare and assign separately to avoid masking return values.


In scripts/load-secrets.sh line 10:
    export POSTGRES_PASSWORD=$(cat .secrets/db_password.txt)
           ^---------------^ SC2155 (warning): Declare and assign separately to avoid masking return values.


In scripts/load-secrets.sh line 14:
    export REDIS_PASSWORD=$(cat .secrets/redis_password.txt)
           ^------------^ SC2155 (warning): Declare and assign separately to avoid masking return values.


In scripts/load-secrets.sh line 18:
    export OPENAI_API_KEY=$(cat .secrets/api_key_openai.txt)
           ^------------^ SC2155 (warning): Declare and assign separately to avoid masking return values.


In scripts/load-secrets.sh line 22:
    export ANTHROPIC_API_KEY=$(cat .secrets/api_key_anthropic.txt)
           ^---------------^ SC2155 (warning): Declare and assign separately to avoid masking return values.


In scripts/load-secrets.sh line 26:
    export GOOGLE_API_KEY=$(cat .secrets/api_key_google.txt)
           ^------------^ SC2155 (warning): Declare and assign separately to avoid masking return values.


In scripts/load-secrets.sh line 30:
    export SESSION_SECRET=$(cat .secrets/session_secret.txt)
           ^------------^ SC2155 (warning): Declare and assign separately to avoid masking return values.


In scripts/setup_postgres.sh line 100:
for i in {1..30}; do
^-^ SC2034 (warning): i appears unused. Verify use (or export if used externally).

For more information:
  https://www.shellcheck.net/wiki/SC2218 -- This function is only defined lat...
  https://www.shellcheck.net/wiki/SC2034 -- BLUE appears unused. Verify use (...
  https://www.shellcheck.net/wiki/SC2046 -- Quote this to prevent word splitt...

Hadolint Dockerfile linter...............................................Failed
- hook id: hadolint-docker
- exit code: 125

Unable to find image 'hadolint:latest' locally
docker: Error response from daemon: pull access denied for hadolint, repository does not exist or may require 'docker login'

Run 'docker run --help' for more information
Unable to find image 'hadolint:latest' locally
docker: Error response from daemon: pull access denied for hadolint, repository does not exist or may require 'docker login'

Run 'docker run --help' for more information
Unable to find image 'hadolint:latest' locally
docker: Error response from daemon: pull access denied for hadolint, repository does not exist or may require 'docker login'

Run 'docker run --help' for more information

Validate GitHub workflows................................................Passed
Validate GitHub actions..............................(no files to check)Skipped
ESLint (frontend)....................................(no files to check)Skipped
Prettier (frontend)......................................................Passed
SQLFluff lint............................................................Failed
- hook id: sqlfluff-lint
- exit code: 1

WARNING    Length of file 'scripts/bot-sessions-schema.sql' is 22079 bytes which is over the limit of 20000 bytes. Skipping to avoid parser lock. Users can increase this limit in their config by setting the 'large_file_skip_byte_limit' value, or disable by setting it to zero. 
WARNING    Length of file 'scripts/database-init-complete.sql' is 28656 bytes which is over the limit of 20000 bytes. Skipping to avoid parser lock. Users can increase this limit in their config by setting the 'large_file_skip_byte_limit' value, or disable by setting it to zero. 
== [scripts/migrations/001_speaker_enhancements.sql] FAIL
L:  28 | P:   5 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L:  28 | P:  70 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  30 | P:   5 | LT05 | Line is too long (85 > 80). [layout.long_lines]
L:  30 | P:  69 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  32 | P:   5 | LT05 | Line is too long (105 > 80). [layout.long_lines]
L:  35 | P:  34 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L:  40 | P:  11 | LT01 | Expected single whitespace between 'UNIQUE' keyword and
                       | start bracket '('. [layout.spacing]
L:  45 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  45 | P:  39 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  48 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  48 | P:  39 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  51 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  51 | P:  39 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  54 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  54 | P:  39 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  96 | P:   1 | LT05 | Line is too long (84 > 80). [layout.long_lines]
L:  98 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  99 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  99 | P:   5 | LT05 | Line is too long (81 > 80). [layout.long_lines]
L: 111 | P:   1 | LT05 | Line is too long (86 > 80). [layout.long_lines]
L: 113 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 114 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 114 | P:   5 | LT05 | Line is too long (82 > 80). [layout.long_lines]
L: 118 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 118 | P:  39 | CP02 | Unquoted identifiers must be consistently lower case.
                       | [capitalisation.identifiers]
L: 118 | P:  42 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 121 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 121 | P:  39 | CP02 | Unquoted identifiers must be consistently lower case.
                       | [capitalisation.identifiers]
L: 121 | P:  42 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 124 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 124 | P:  40 | CP02 | Unquoted identifiers must be consistently lower case.
                       | [capitalisation.identifiers]
L: 124 | P:  43 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 128 | P:  21 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 132 | P:  21 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 172 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 172 | P:  32 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 175 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 175 | P:  32 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 178 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 178 | P:  32 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 186 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 186 | P:  32 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 194 | P:   1 | ST06 | Select wildcards then simple targets before calculations
                       | and aggregates. [structure.column_order]
L: 197 | P:  50 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 200 | P:  14 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 201 | P:  46 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 202 | P:  29 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 203 | P:  40 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 204 | P:  39 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 205 | P:  31 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 206 | P:  43 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 208 | P:  37 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 211 | P:   9 | LT02 | Expected line break and indent of 4 spaces before 't'.
                       | [layout.indent]
L: 212 | P:   1 | LT02 | Expected indent of 4 spaces. [layout.indent]
L: 219 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 219 | P:   5 | LT05 | Line is too long (96 > 80). [layout.long_lines]
L: 222 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 225 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 228 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 231 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 231 | P:   5 | LT05 | Line is too long (84 > 80). [layout.long_lines]
== [scripts/chat-history-schema.sql] FAIL
L:  15 | P:  32 | RF06 | Unnecessary quoted identifier "pg_trgm".
                       | [references.quoting]
L:  35 | P:  46 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L:  47 | P:  51 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  48 | P:  56 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  49 | P:  57 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  50 | P:   1 | LT05 | Line is too long (84 > 80). [layout.long_lines]
L:  50 | P:  71 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  53 | P:  28 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L:  62 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  63 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  64 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L:  64 | P:  22 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L:  72 | P:  39 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L:  76 | P:  51 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  80 | P:  56 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L:  92 | P:  61 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  93 | P:  63 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  94 | P:  66 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 102 | P:  41 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 105 | P:  51 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 129 | P:   1 | LT05 | Line is too long (86 > 80). [layout.long_lines]
L: 129 | P:  77 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 130 | P:   1 | LT05 | Line is too long (92 > 80). [layout.long_lines]
L: 130 | P:  80 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 131 | P:   1 | LT05 | Line is too long (102 > 80). [layout.long_lines]
L: 131 | P:  85 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 132 | P:   1 | LT05 | Line is too long (96 > 80). [layout.long_lines]
L: 132 | P:  82 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 133 | P:   1 | LT05 | Line is too long (103 > 80). [layout.long_lines]
L: 133 | P:  82 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 134 | P:   1 | LT05 | Line is too long (110 > 80). [layout.long_lines]
L: 134 | P:  92 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 142 | P:  41 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 145 | P:   5 | LT05 | Line is too long (92 > 80). [layout.long_lines]
L: 145 | P:  62 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 177 | P:   1 | LT05 | Line is too long (84 > 80). [layout.long_lines]
L: 177 | P:  72 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 178 | P:   1 | LT05 | Line is too long (82 > 80). [layout.long_lines]
L: 178 | P:  71 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 179 | P:  66 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 180 | P:   1 | LT05 | Line is too long (99 > 80). [layout.long_lines]
L: 180 | P:  70 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 181 | P:   1 | LT05 | Line is too long (110 > 80). [layout.long_lines]
L: 181 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 182 | P:   1 | LT05 | Line is too long (106 > 80). [layout.long_lines]
L: 182 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 185 | P:  28 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 199 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 200 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 201 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 201 | P:  22 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 204 | P:  28 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 218 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 219 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 220 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 220 | P:  22 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 228 | P:  44 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 229 | P:   5 | LT05 | Line is too long (99 > 80). [layout.long_lines]
L: 229 | P:  69 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 250 | P:   5 | LT05 | Line is too long (93 > 80). [layout.long_lines]
L: 257 | P:   1 | LT05 | Line is too long (91 > 80). [layout.long_lines]
L: 257 | P:  79 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 258 | P:   1 | LT05 | Line is too long (97 > 80). [layout.long_lines]
L: 258 | P:  82 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 275 | P:  12 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 276 | P:   1 | AM05 | Join clauses should be fully qualified. [ambiguous.join]
L: 276 | P:  28 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 289 | P:  26 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 290 | P:  23 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 291 | P:  28 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 292 | P:  25 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 303 | P:  35 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 304 | P:  26 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 305 | P:  29 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 306 | P:  27 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 307 | P:  12 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 308 | P:  33 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 309 | P:  25 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 318 | P:  28 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 319 | P:   9 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 404 | P:  28 | CP03 | Function names must be consistently upper case.
                       | [capitalisation.functions]
L: 467 | P:   1 | LT05 | Line is too long (92 > 80). [layout.long_lines]
L: 476 | P:  54 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 480 | P:  13 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 481 | P:  14 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 484 | P:   1 | LT09 | Select targets should be on a new line unless there is
                       | only one select target. [layout.select_targets]
L: 484 | P:   8 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
L: 484 | P:  33 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
L: 486 | P:   1 | LT09 | Select targets should be on a new line unless there is
                       | only one select target. [layout.select_targets]
L: 486 | P:   8 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
L: 486 | P:  25 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
L: 488 | P:   1 | LT09 | Select targets should be on a new line unless there is
                       | only one select target. [layout.select_targets]
L: 488 | P:   8 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
L: 488 | P:  22 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
L: 490 | P:   1 | LT09 | Select targets should be on a new line unless there is
                       | only one select target. [layout.select_targets]
L: 490 | P:   8 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
L: 490 | P:  35 | AL03 | Column expression without alias. Use explicit `AS`
                       | clause. [aliasing.expression]
== [scripts/migration-translation-optimization.sql] FAIL
L:   2 | P:   1 | LT05 | Line is too long (91 > 80). [layout.long_lines]
L:  13 | P:   5 | LT05 | Line is too long (91 > 80). [layout.long_lines]
L:  13 | P:  61 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  24 | P:   5 | LT05 | Line is too long (86 > 80). [layout.long_lines]
L:  38 | P:  34 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L:  45 | P:   5 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L:  45 | P:  70 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  50 | P:  26 | LT01 | Expected single whitespace between 'TEXT' keyword and
                       | start square bracket '['. [layout.spacing]
L:  65 | P:   5 | LT05 | Line is too long (92 > 80). [layout.long_lines]
L:  77 | P:  34 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L:  87 | P:   5 | LT05 | Line is too long (89 > 80). [layout.long_lines]
L: 121 | P:  34 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 124 | P:   5 | LT05 | Line is too long (107 > 80). [layout.long_lines]
L: 131 | P:   5 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 131 | P:  70 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 138 | P:   5 | LT05 | Line is too long (96 > 80). [layout.long_lines]
L: 138 | P:  45 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 141 | P:   5 | LT05 | Line is too long (81 > 80). [layout.long_lines]
L: 141 | P:  44 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 144 | P:  43 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 150 | P:  34 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 159 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 160 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 161 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 161 | P:   5 | LT05 | Line is too long (124 > 80). [layout.long_lines]
L: 161 | P:  95 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 162 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 163 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 164 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 165 | P:   1 | LT02 | Line should not be indented. [layout.indent]
L: 165 | P:  72 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 168 | P:   1 | LT05 | Line is too long (124 > 80). [layout.long_lines]
L: 175 | P:   1 | LT05 | Line is too long (106 > 80). [layout.long_lines]
L: 175 | P:  94 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 176 | P:   1 | LT05 | Line is too long (109 > 80). [layout.long_lines]
L: 176 | P:  93 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 177 | P:   1 | LT05 | Line is too long (102 > 80). [layout.long_lines]
L: 177 | P:  87 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 178 | P:   1 | LT05 | Line is too long (127 > 80). [layout.long_lines]
L: 178 | P:  93 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 179 | P:   1 | LT05 | Line is too long (106 > 80). [layout.long_lines]
L: 179 | P:  94 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 182 | P:   1 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 182 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 183 | P:   1 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 183 | P:  81 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 184 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 184 | P:  84 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 185 | P:   1 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 185 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 188 | P:   1 | LT05 | Line is too long (94 > 80). [layout.long_lines]
L: 188 | P:  82 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 189 | P:   1 | LT05 | Line is too long (120 > 80). [layout.long_lines]
L: 189 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 190 | P:   1 | LT05 | Line is too long (109 > 80). [layout.long_lines]
L: 190 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 191 | P:   1 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 191 | P:  84 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 194 | P:   1 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 194 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 195 | P:   1 | LT05 | Line is too long (104 > 80). [layout.long_lines]
L: 195 | P:  87 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 196 | P:   1 | LT05 | Line is too long (99 > 80). [layout.long_lines]
L: 196 | P:  84 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 199 | P:   1 | LT05 | Line is too long (91 > 80). [layout.long_lines]
L: 199 | P:  79 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 200 | P:   1 | LT05 | Line is too long (92 > 80). [layout.long_lines]
L: 200 | P:  80 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 201 | P:   1 | LT05 | Line is too long (92 > 80). [layout.long_lines]
L: 201 | P:  82 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 204 | P:   1 | LT05 | Line is too long (124 > 80). [layout.long_lines]
L: 204 | P:  97 | CP02 | Unquoted identifiers must be consistently lower case.
                       | [capitalisation.identifiers]
L: 217 | P:  14 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 218 | P:  52 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 219 | P:  56 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 220 | P:   5 | LT05 | Line is too long (120 > 80). [layout.long_lines]
L: 220 | P:  60 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 220 | P: 101 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 221 | P:   5 | LT05 | Line is too long (107 > 80). [layout.long_lines]
L: 221 | P:  15 | ST01 | Do not specify 'else null' in a case when statement
                       | (redundant). [structure.else_null]
L: 221 | P:  66 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 221 | P:  80 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 222 | P:   5 | LT05 | Line is too long (119 > 80). [layout.long_lines]
L: 222 | P:  15 | ST01 | Do not specify 'else null' in a case when statement
                       | (redundant). [structure.else_null]
L: 222 | P:  76 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 222 | P:  90 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 223 | P:  37 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 232 | P:  14 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 233 | P:  34 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 234 | P:  45 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 235 | P:  44 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 236 | P:  41 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 237 | P:  26 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 238 | P:  27 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 239 | P:   5 | LT05 | Line is too long (117 > 80). [layout.long_lines]
L: 239 | P:  31 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 239 | P:  95 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 249 | P:  29 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 250 | P:  35 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 251 | P:  35 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 252 | P:  35 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 253 | P:  41 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 254 | P:   5 | LT05 | Line is too long (118 > 80). [layout.long_lines]
L: 254 | P:  41 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 254 | P:  96 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 263 | P:  38 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 264 | P:  39 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 265 | P:  34 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 266 | P:  51 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 267 | P:   5 | LT05 | Line is too long (125 > 80). [layout.long_lines]
L: 267 | P:  59 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 267 | P: 100 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 268 | P:  45 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 269 | P:   5 | LT05 | Line is too long (118 > 80). [layout.long_lines]
L: 269 | P: 101 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 270 | P:  32 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 271 | P:   5 | LT05 | Line is too long (91 > 80). [layout.long_lines]
L: 271 | P:  72 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 271 | P:  78 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 272 | P:  32 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 288 | P:  43 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 289 | P:  34 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 290 | P:  43 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 291 | P:  31 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 292 | P:  28 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 319 | P:  28 | LT01 | Expected single whitespace between 'TEXT' keyword and
                       | start square bracket '['. [layout.spacing]
L: 323 | P:  39 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
L: 326 | P:  35 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 364 | P:  31 | CP04 | Boolean/null literals must be consistently lower case.
                       | [capitalisation.literals]
== [scripts/init-db.sql] FAIL
L:   6 | P:  32 | RF06 | Unnecessary quoted identifier "pg_trgm".
                       | [references.quoting]
L:  21 | P:  33 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  25 | P:  50 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L:  30 | P:  32 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  31 | P:  30 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  37 | P:  60 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  41 | P:  26 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  46 | P:  34 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  47 | P:  32 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  48 | P:  30 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  51 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L:  52 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L:  58 | P:  60 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  64 | P:  26 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  69 | P:  41 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L:  70 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L:  71 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L:  79 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L:  81 | P:  52 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L:  87 | P:  65 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L:  91 | P:  26 | LT01 | Expected single whitespace between 'TEXT' keyword and
                       | start square bracket '['. [layout.spacing]
L:  92 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 111 | P:  30 | LT01 | Expected single whitespace between 'TEXT' keyword and
                       | start square bracket '['. [layout.spacing]
L: 112 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 113 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 123 | P:  64 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 126 | P:  48 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 139 | P:  64 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 140 | P:  60 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 141 | P:  60 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 142 | P:  32 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 143 | P:  30 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 144 | P:  33 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 145 | P:  26 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 146 | P:  36 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 147 | P:  34 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 149 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 155 | P:  66 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 156 | P:  64 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 158 | P:  48 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 166 | P:  64 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 169 | P:  40 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 173 | P:  34 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 174 | P:  32 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 176 | P:  28 | LT01 | Expected single whitespace between 'TEXT' keyword and
                       | start square bracket '['. [layout.spacing]
L: 177 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 199 | P:  26 | LT01 | Expected single whitespace between 'TEXT' keyword and
                       | start square bracket '['. [layout.spacing]
L: 204 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 205 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 212 | P:  54 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 226 | P:  49 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 234 | P:  29 | LT01 | Expected single whitespace between comma ',' and numeric
                       | literal. [layout.spacing]
L: 236 | P:  48 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 243 | P:   1 | LT05 | Line is too long (87 > 80). [layout.long_lines]
L: 243 | P:  79 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 244 | P:   1 | LT05 | Line is too long (97 > 80). [layout.long_lines]
L: 244 | P:  84 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 245 | P:   1 | LT05 | Line is too long (95 > 80). [layout.long_lines]
L: 245 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 246 | P:   1 | LT05 | Line is too long (115 > 80). [layout.long_lines]
L: 246 | P:  87 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 247 | P:   1 | LT05 | Line is too long (115 > 80). [layout.long_lines]
L: 247 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 249 | P:   1 | LT05 | Line is too long (101 > 80). [layout.long_lines]
L: 249 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 250 | P:   1 | LT05 | Line is too long (95 > 80). [layout.long_lines]
L: 250 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 251 | P:   1 | LT05 | Line is too long (91 > 80). [layout.long_lines]
L: 251 | P:  81 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 252 | P:   1 | LT05 | Line is too long (95 > 80). [layout.long_lines]
L: 252 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 253 | P:   1 | LT05 | Line is too long (115 > 80). [layout.long_lines]
L: 253 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 254 | P:   1 | LT05 | Line is too long (95 > 80). [layout.long_lines]
L: 254 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 257 | P:   1 | LT05 | Line is too long (101 > 80). [layout.long_lines]
L: 257 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 258 | P:   1 | LT05 | Line is too long (95 > 80). [layout.long_lines]
L: 258 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 259 | P:   1 | LT05 | Line is too long (116 > 80). [layout.long_lines]
L: 259 | P:  82 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 260 | P:   1 | LT05 | Line is too long (95 > 80). [layout.long_lines]
L: 260 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 263 | P:   1 | LT05 | Line is too long (92 > 80). [layout.long_lines]
L: 263 | P:  77 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 264 | P:   1 | LT05 | Line is too long (91 > 80). [layout.long_lines]
L: 264 | P:  79 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 265 | P:   1 | LT05 | Line is too long (88 > 80). [layout.long_lines]
L: 265 | P:  79 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 266 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 266 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 267 | P:   1 | LT05 | Line is too long (117 > 80). [layout.long_lines]
L: 267 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 268 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 268 | P:  84 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 269 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 269 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 270 | P:   1 | LT05 | Line is too long (104 > 80). [layout.long_lines]
L: 270 | P:  87 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 273 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 273 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 274 | P:   1 | LT05 | Line is too long (90 > 80). [layout.long_lines]
L: 274 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 275 | P:   1 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 275 | P:  87 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 277 | P:   1 | LT05 | Line is too long (108 > 80). [layout.long_lines]
L: 277 | P:  96 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 278 | P:   1 | LT05 | Line is too long (108 > 80). [layout.long_lines]
L: 278 | P:  96 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 279 | P:   1 | LT05 | Line is too long (115 > 80). [layout.long_lines]
L: 279 | P:  92 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 281 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 281 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 282 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 282 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 283 | P:   1 | LT05 | Line is too long (120 > 80). [layout.long_lines]
L: 283 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 284 | P:   1 | LT05 | Line is too long (103 > 80). [layout.long_lines]
L: 284 | P:  88 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 286 | P:   1 | LT05 | Line is too long (108 > 80). [layout.long_lines]
L: 286 | P:  96 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 287 | P:   1 | LT05 | Line is too long (120 > 80). [layout.long_lines]
L: 287 | P:  94 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 288 | P:   1 | LT05 | Line is too long (106 > 80). [layout.long_lines]
L: 288 | P:  95 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 290 | P:   1 | LT05 | Line is too long (102 > 80). [layout.long_lines]
L: 290 | P:  90 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 291 | P:   1 | LT05 | Line is too long (102 > 80). [layout.long_lines]
L: 291 | P:  90 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 292 | P:   1 | LT05 | Line is too long (130 > 80). [layout.long_lines]
L: 292 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 295 | P:   1 | LT05 | Line is too long (97 > 80). [layout.long_lines]
L: 295 | P:  83 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 296 | P:   1 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 296 | P:  86 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 297 | P:   1 | LT05 | Line is too long (120 > 80). [layout.long_lines]
L: 297 | P:  93 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 298 | P:   1 | LT05 | Line is too long (106 > 80). [layout.long_lines]
L: 298 | P:  95 | LT01 | Expected single whitespace between naked identifier and
                       | start bracket '('. [layout.spacing]
L: 315 | P:  21 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 316 | P:  13 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 317 | P:  32 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 318 | P:   1 | AM05 | Join clauses should be fully qualified. [ambiguous.join]
L: 318 | P:  32 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 319 | P:  37 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 320 | P:  23 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 334 | P:   9 | LT02 | Expected line break and indent of 8 spaces before
                       | 'WHEN'. [layout.indent]
L: 335 | P:   1 | LT02 | Expected indent of 12 spaces. [layout.indent]
L: 335 | P:   9 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 335 | P:  29 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 336 | P:   1 | LT02 | Expected indent of 8 spaces. [layout.indent]
L: 337 | P:   1 | LT02 | Expected indent of 12 spaces. [layout.indent]
L: 337 | P:   9 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 338 | P:   9 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 339 | P:  36 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 340 | P:  37 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 354 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 354 | P:  18 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 355 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 355 | P:  35 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 356 | P:  32 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 357 | P:   1 | AM05 | Join clauses should be fully qualified. [ambiguous.join]
L: 357 | P:  29 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 357 | P:  50 | CP05 | Datatypes must be consistently upper case.
                       | [capitalisation.types]
L: 358 | P:   1 | LT05 | Line is too long (99 > 80). [layout.long_lines]
L: 358 | P:  41 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 358 | P:  96 | CP04 | Boolean/null literals must be consistently upper case.
                       | [capitalisation.literals]
L: 360 | P:   9 | LT02 | Expected line break and indent of 4 spaces before 'ms'.
                       | [layout.indent]
L: 361 | P:   1 | LT02 | Expected indent of 4 spaces. [layout.indent]
L: 361 | P:  10 | LT05 | Line is too long (98 > 80). [layout.long_lines]
L: 368 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 368 | P:  35 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 369 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 369 | P:  33 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 370 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 370 | P:  30 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 371 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 371 | P:  32 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 372 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 372 | P:  30 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 373 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 373 | P:  32 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 374 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 374 | P:  24 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 375 | P:  33 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 376 | P:  37 | AL01 | Implicit/explicit aliasing of table. [aliasing.table]
L: 384 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 384 | P:  14 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 385 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 385 | P:  27 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 386 | P:   5 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 386 | P:  21 | CP01 | Keywords must be consistently upper case.
                       | [capitalisation.keywords]
L: 388 | P:  21 | CP03 | Function names must be consistently lower case.
                       | [capitalisation.functions]
L: 393 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 420 | P:   1 | LT05 | Line is too long (100 > 80). [layout.long_lines]
L: 421 | P:   1 | LT05 | Line is too long (84 > 80). [layout.long_lines]
All Finished!
