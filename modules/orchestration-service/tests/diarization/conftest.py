"""
Diarization test conftest.

Pipeline and auto-trigger tests exercise the real database-backed
DiarizationPipeline via the shared testcontainer infrastructure provided
by the global conftest (postgres container, Alembic migrations, etc.).

Tests that only need Pydantic models (test_diarization_models.py,
test_diarization_rules.py, etc.) do not touch the DB at all — they just
import models and call pure-Python functions, so the DB fixtures are
simply unused in those cases without any need to stub them out.
"""
