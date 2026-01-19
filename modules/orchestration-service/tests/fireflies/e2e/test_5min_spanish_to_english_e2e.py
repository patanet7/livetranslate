#!/usr/bin/env python3
"""
5-Minute End-to-End Test: Spanish → English Translation with OBS Caption Output

This test verifies the complete pipeline:
1. Mock Fireflies server streams Spanish transcript chunks
2. Orchestration service processes chunks via TranscriptionPipelineCoordinator
3. Real translation service (V3 API) translates Spanish → English
4. Captions broadcast to WebSocket for OBS Browser Source display

To run with visual output in OBS:
1. Start orchestration service: python src/main_fastapi.py
2. Start translation service: cd ../translation-service && python src/api_server_fastapi.py
3. Open OBS and add Browser Source: http://localhost:3000/static/captions.html?session=test-e2e
4. Run this test: python tests/fireflies/e2e/test_5min_spanish_to_english_e2e.py

Results saved to: tests/output/TIMESTAMP_test_5min_e2e_results.log
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Test configuration
MOCK_SERVER_PORT = 8096
ORCHESTRATION_URL = "http://localhost:3000"
TRANSLATION_URL = "http://localhost:5003"
TEST_SESSION_ID = "test-e2e"
TEST_API_KEY = "test-fireflies-e2e"

# Output file
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SpanishTranscriptEntry:
    """Spanish transcript entry with expected English translation."""

    speaker: str
    spanish_text: str
    timestamp_ms: int
    expected_english: str


# 5-minute Spanish meeting transcript (~60 sentences, 5 seconds apart)
# Technical meeting discussion in Spanish with expected English translations
SPANISH_MEETING_TRANSCRIPT: list[SpanishTranscriptEntry] = [
    # Opening (0-30 seconds)
    SpanishTranscriptEntry(
        "Carlos",
        "Buenos días a todos. Vamos a discutir los cambios de la API.",
        0,
        "Good morning everyone. Let's discuss the API changes.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Gracias Carlos. He revisado la documentación del punto de acceso.",
        5000,
        "Thank you Carlos. I have reviewed the endpoint documentation.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Tengo algunas preocupaciones sobre el proceso de despliegue.",
        10000,
        "I have some concerns about the deployment process.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Escuchemos. La arquitectura de microservicio necesita planificación.",
        15000,
        "Let's hear them. The microservice architecture needs planning.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Definitivamente deberíamos planificar escenarios de reversión.",
        20000,
        "We should definitely plan for rollback scenarios.",
    ),
    SpanishTranscriptEntry(
        "María",
        "De acuerdo. ¿Qué hay de las migraciones de base de datos?",
        25000,
        "Agreed. What about the database migrations?",
    ),
    # Technical discussion (30-90 seconds)
    SpanishTranscriptEntry(
        "Carlos",
        "El servidor necesita manejar el nuevo flujo de autenticación.",
        30000,
        "The server needs to handle the new authentication flow.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Configuraré los ajustes de caché para mejor rendimiento.",
        35000,
        "I will configure the cache settings for better performance.",
    ),
    SpanishTranscriptEntry(
        "María",
        "El equipo de cliente está esperando las especificaciones.",
        40000,
        "The client team is waiting for the specifications.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Necesitamos probar cada punto de acceso exhaustivamente.",
        45000,
        "We need to test each endpoint thoroughly.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "El pipeline de despliegue está casi listo.",
        50000,
        "The deployment pipeline is almost ready.",
    ),
    SpanishTranscriptEntry(
        "María",
        "¿Cuándo podemos programar la próxima revisión?",
        55000,
        "When can we schedule the next review?",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Planeemos para el viernes con todo el equipo.",
        60000,
        "Let's plan for Friday with the whole team.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Prepararé las métricas del servidor para entonces.",
        65000,
        "I will prepare the server metrics by then.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Los requisitos del cliente han cambiado un poco.",
        70000,
        "The client requirements have changed a bit.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "¿Cómo afecta eso a nuestro diseño de la API?",
        75000,
        "How does that affect our API design?",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Podríamos necesitar actualizar el módulo de autenticación.",
        80000,
        "We might need to update the authentication module.",
    ),
    SpanishTranscriptEntry(
        "María",
        "La estrategia de caché debería permanecer igual.",
        85000,
        "The cache strategy should remain the same.",
    ),
    # Deep dive (90-150 seconds)
    SpanishTranscriptEntry(
        "Carlos",
        "Revisemos los requisitos de seguridad del sistema.",
        90000,
        "Let's review the system security requirements.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Las reglas del firewall están configuradas correctamente.",
        95000,
        "The firewall rules are configured correctly.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Nuestros socios necesitan acceso temprano para probar.",
        100000,
        "Our partners need early access to test.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Podemos proporcionar un entorno de pruebas separado.",
        105000,
        "We can provide a separate test environment.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Me encargaré del despliegue al servidor de pruebas.",
        110000,
        "I will handle the deployment to the test server.",
    ),
    SpanishTranscriptEntry(
        "María",
        "¿Cuál es el plan si algo sale mal?",
        115000,
        "What is the plan if something goes wrong?",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "El microservicio puede revertirse independientemente.",
        120000,
        "The microservice can be rolled back independently.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "El esquema de base de datos soporta compatibilidad.",
        125000,
        "The database schema supports compatibility.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Perfecto. El equipo de desarrollo lo apreciará.",
        130000,
        "Perfect. The development team will appreciate it.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Documentemos el proceso de autenticación claramente.",
        135000,
        "Let's document the authentication process clearly.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Actualizaré la guía de configuración del servidor.",
        140000,
        "I will update the server configuration guide.",
    ),
    SpanishTranscriptEntry(
        "María",
        "La documentación de caché también necesita actualización.",
        145000,
        "The cache documentation also needs updating.",
    ),
    # Planning (150-210 seconds)
    SpanishTranscriptEntry(
        "Carlos",
        "La estrategia de versionado necesita más discusión.",
        150000,
        "The versioning strategy needs more discussion.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Deberíamos usar versionado semántico para todo.",
        155000,
        "We should use semantic versioning for everything.",
    ),
    SpanishTranscriptEntry(
        "María",
        "El SDK del cliente necesitará actualizaciones también.",
        160000,
        "The client SDK will need updates as well.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Planifiquemos el despliegue en varias fases.",
        165000,
        "Let's plan the deployment in several phases.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "La fase uno cubre los cambios principales del sistema.",
        170000,
        "Phase one covers the main system changes.",
    ),
    SpanishTranscriptEntry(
        "María",
        "¿Cuándo ocurre la migración de la base de datos?",
        175000,
        "When does the database migration happen?",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "El equipo de infraestructura se encarga en la fase dos.",
        180000,
        "The infrastructure team handles it in phase two.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Necesitamos pruebas exhaustivas antes de producción.",
        185000,
        "We need thorough testing before production.",
    ),
    SpanishTranscriptEntry(
        "María",
        "El equipo ejecutará todas las pruebas de integración.",
        190000,
        "The team will run all integration tests.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Bien. Finalicemos el contrato de la API hoy.",
        195000,
        "Good. Let's finalize the API contract today.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Los tokens necesitan tiempos de expiración más largos.",
        200000,
        "The tokens need longer expiration times.",
    ),
    SpanishTranscriptEntry(
        "María",
        "El tiempo de espera del caché debería coincidir.",
        205000,
        "The cache timeout should match.",
    ),
    # Implementation details (210-270 seconds)
    SpanishTranscriptEntry(
        "Carlos",
        "Revisemos los límites de velocidad del sistema.",
        210000,
        "Let's review the system rate limits.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "El servidor puede manejar mil solicitudes por segundo.",
        215000,
        "The server can handle a thousand requests per second.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Nuestras pruebas de carga muestran números similares.",
        220000,
        "Our load tests show similar numbers.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "La puerta de enlace manejará el desbordamiento bien.",
        225000,
        "The gateway will handle the overflow well.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Configuraré el panel de monitoreo del despliegue.",
        230000,
        "I will set up the deployment monitoring dashboard.",
    ),
    SpanishTranscriptEntry(
        "María",
        "¿Puede el cliente recibir actualizaciones en tiempo real?",
        235000,
        "Can the client receive real-time updates?",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Sí, a través del flujo de eventos del microservicio.",
        240000,
        "Yes, through the microservice event stream.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "El pool de conexiones está optimizado correctamente.",
        245000,
        "The connection pool is properly optimized.",
    ),
    SpanishTranscriptEntry(
        "María",
        "¿Qué hay del flujo de actualización de tokens?",
        250000,
        "What about the token refresh flow?",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "El servidor maneja eso automáticamente ahora.",
        255000,
        "The server handles that automatically now.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "La invalidación de caché está sincronizada entre nodos.",
        260000,
        "Cache invalidation is synchronized across nodes.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Esa es una mejora importante para los usuarios.",
        265000,
        "That is an important improvement for users.",
    ),
    # Closing (270-300 seconds)
    SpanishTranscriptEntry(
        "Carlos",
        "Resumamos los cambios que discutimos hoy.",
        270000,
        "Let's summarize the changes we discussed today.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "El cronograma de despliegue parece alcanzable.",
        275000,
        "The deployment schedule seems achievable.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Informaré al equipo sobre el cronograma completo.",
        280000,
        "I will inform the team about the complete schedule.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "La documentación debería estar lista mañana.",
        285000,
        "The documentation should be ready tomorrow.",
    ),
    SpanishTranscriptEntry(
        "Pedro",
        "Completaré la configuración del servidor hoy.",
        290000,
        "I will complete the server configuration today.",
    ),
    SpanishTranscriptEntry(
        "María",
        "Gran reunión todos. Gracias por la discusión.",
        295000,
        "Great meeting everyone. Thanks for the discussion.",
    ),
    SpanishTranscriptEntry(
        "Carlos",
        "Gracias. Nos reunimos el viernes con actualizaciones.",
        300000,
        "Thank you. We meet Friday with updates.",
    ),
]


# Spanish glossary terms for technical consistency
SPANISH_GLOSSARY: dict[str, str] = {
    "API": "API",
    "punto de acceso": "endpoint",
    "microservicio": "microservice",
    "despliegue": "deployment",
    "reversión": "rollback",
    "base de datos": "database",
    "servidor": "server",
    "cliente": "client",
    "autenticación": "authentication",
    "caché": "cache",
}


class TestResults:
    """Collects test results for logging."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.translations = []
        self.start_time = datetime.now()

    def add_pass(self, name: str, details: str = ""):
        self.passed.append({"name": name, "details": details})
        logger.info(f"[PASS] {name}: {details}")

    def add_fail(self, name: str, error: str):
        self.failed.append({"name": name, "error": error})
        logger.error(f"[FAIL] {name}: {error}")

    def add_translation(self, entry: SpanishTranscriptEntry, translation: str, time_ms: int):
        self.translations.append(
            {
                "speaker": entry.speaker,
                "original": entry.spanish_text,
                "translated": translation,
                "expected": entry.expected_english,
                "time_ms": time_ms,
            }
        )

    def write_log(self, filepath: Path):
        with open(filepath, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("5-Minute Spanish→English E2E Test Results\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {self.start_time.isoformat()}\n")
            f.write(f"Duration: {(datetime.now() - self.start_time).total_seconds():.1f}s\n")
            f.write(f"Passed: {len(self.passed)}\n")
            f.write(f"Failed: {len(self.failed)}\n")
            f.write(f"Translations: {len(self.translations)}\n")
            f.write("=" * 70 + "\n\n")

            f.write("--- Test Results ---\n\n")
            for p in self.passed:
                f.write(f"[PASS] {p['name']}\n")
                if p["details"]:
                    f.write(f"       {p['details']}\n")

            for fail in self.failed:
                f.write(f"[FAIL] {fail['name']}\n")
                f.write(f"       Error: {fail['error']}\n")

            f.write("\n--- Translations ---\n\n")
            for t in self.translations:
                f.write(f"[{t['speaker']}] ({t['time_ms']}ms)\n")
                f.write(f"  ES: {t['original']}\n")
                f.write(f"  EN: {t['translated']}\n")
                f.write(f"  Expected: {t['expected']}\n\n")

        return filepath


async def check_services() -> dict[str, bool]:
    """Check if required services are running."""
    results = {}

    async with aiohttp.ClientSession() as session:
        # Check orchestration service (uses /api/health endpoint)
        try:
            async with session.get(
                f"{ORCHESTRATION_URL}/api/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                results["orchestration"] = resp.status == 200
        except Exception:
            results["orchestration"] = False

        # Check translation service
        try:
            async with session.get(
                f"{TRANSLATION_URL}/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                results["translation"] = resp.status == 200
        except Exception:
            results["translation"] = False

    return results


async def translate_text(
    session: aiohttp.ClientSession, text: str, glossary: dict[str, str]
) -> dict[str, Any]:
    """Call translation service V3 API for Spanish → English."""
    url = f"{TRANSLATION_URL}/api/v3/translate"

    # Build glossary instruction if terms provided
    glossary_instruction = ""
    if glossary:
        glossary_lines = [f'  "{k}" -> "{v}"' for k, v in list(glossary.items())[:5]]
        glossary_instruction = "\n\nUse these technical term translations:\n" + "\n".join(
            glossary_lines
        )

    # Build the prompt in the format V3 API expects
    prompt = f"""Translate this Spanish text to English. Return ONLY the English translation, nothing else.{glossary_instruction}

Spanish: {text}

English:"""

    payload = {
        "prompt": prompt,
        "backend": "ollama",
        "max_tokens": 256,
        "temperature": 0.3,
        "system_prompt": "You are a professional translator. Return ONLY the translation, no explanations.",
    }

    start = datetime.now()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                result = await resp.json()
                elapsed_ms = int((datetime.now() - start).total_seconds() * 1000)
                translated = result.get("text", "").strip()
                # Clean up any leading "English:" or similar prefixes
                if translated.lower().startswith("english:"):
                    translated = translated[8:].strip()
                return {
                    "success": True,
                    "translated": translated,
                    "time_ms": elapsed_ms,
                    "glossary_applied": bool(glossary),
                }
            else:
                error = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {error[:100]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_caption_to_websocket(
    session: aiohttp.ClientSession,
    caption_data: dict[str, Any],
) -> bool:
    """Send caption to orchestration service for WebSocket broadcast."""
    session_id = caption_data.get("session_id", TEST_SESSION_ID)
    url = f"{ORCHESTRATION_URL}/api/captions/{session_id}"

    # Format for AddCaptionRequest schema
    payload = {
        "text": caption_data.get("translated_text", ""),
        "original_text": caption_data.get("original_text", ""),
        "speaker_name": caption_data.get("speaker_name", "Speaker"),
        "target_language": caption_data.get("target_language", "en"),
        "confidence": caption_data.get("confidence", 0.95),
    }

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 201:
                return True
            else:
                error = await resp.text()
                logger.warning(f"Caption broadcast returned {resp.status}: {error[:100]}")
                return False
    except Exception as e:
        logger.warning(f"Caption broadcast failed: {e}")
        return False


async def run_e2e_test(results: TestResults, speed_multiplier: float = 10.0):
    """
    Run the full E2E test with Spanish → English translation.

    Args:
        results: TestResults collector
        speed_multiplier: How much faster than real-time (10x = 30 sec test for 5 min transcript)
    """
    # Check services
    services = await check_services()
    if not services.get("orchestration"):
        results.add_fail("Service Check", "Orchestration service not running at localhost:3000")
        return
    if not services.get("translation"):
        results.add_fail("Service Check", "Translation service not running at localhost:5003")
        return
    results.add_pass("Service Check", "Both orchestration and translation services healthy")

    # Calculate timing
    total_entries = len(SPANISH_MEETING_TRANSCRIPT)
    real_duration_ms = SPANISH_MEETING_TRANSCRIPT[-1].timestamp_ms
    test_duration_sec = real_duration_ms / 1000.0 / speed_multiplier

    logger.info(
        f"Running {total_entries} entries over {test_duration_sec:.1f}s ({speed_multiplier}x speed)"
    )
    logger.info(
        f"Watch captions at: {ORCHESTRATION_URL}/static/captions.html?session={TEST_SESSION_ID}"
    )

    async with aiohttp.ClientSession() as session:
        translation_count = 0
        caption_count = 0

        for i, entry in enumerate(SPANISH_MEETING_TRANSCRIPT):
            # Calculate delay from previous entry
            if i > 0:
                delay_ms = entry.timestamp_ms - SPANISH_MEETING_TRANSCRIPT[i - 1].timestamp_ms
                delay_sec = delay_ms / 1000.0 / speed_multiplier
                await asyncio.sleep(delay_sec)

            # Translate Spanish → English
            result = await translate_text(session, entry.spanish_text, SPANISH_GLOSSARY)

            if result["success"]:
                translation_count += 1
                translated_text = result["translated"]

                results.add_translation(entry, translated_text, result["time_ms"])

                # Send caption to WebSocket for OBS display
                caption_data = {
                    "session_id": TEST_SESSION_ID,
                    "speaker_name": entry.speaker,
                    "original_text": entry.spanish_text,
                    "translated_text": translated_text,
                    "source_language": "es",
                    "target_language": "en",
                    "confidence": 0.95,
                }

                if await send_caption_to_websocket(session, caption_data):
                    caption_count += 1

                # Log progress
                progress = (i + 1) / total_entries * 100
                logger.info(
                    f"[{progress:5.1f}%] [{entry.speaker}] "
                    f"ES: {entry.spanish_text[:40]}... → "
                    f"EN: {translated_text[:40]}..."
                )
            else:
                results.add_fail(f"Translation {i}", result.get("error", "Unknown error"))

        # Final stats
        results.add_pass(
            "Translation Pipeline", f"{translation_count}/{total_entries} translated successfully"
        )
        results.add_pass(
            "Caption Broadcast", f"{caption_count}/{total_entries} captions sent to WebSocket"
        )


async def main():
    """Main entry point."""
    results = TestResults()

    print("=" * 70)
    print("5-Minute Spanish → English E2E Test")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("  1. Orchestration service: python src/main_fastapi.py")
    print("  2. Translation service: cd ../translation-service && python src/api_server_fastapi.py")
    print()
    print(f"Watch captions at: {ORCHESTRATION_URL}/static/captions.html?session={TEST_SESSION_ID}")
    print()
    print("For OBS: Add Browser Source with above URL")
    print("=" * 70)
    print()

    try:
        await run_e2e_test(results, speed_multiplier=10.0)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        results.add_fail("Test Execution", str(e))
        logger.exception("Test failed with exception")

    # Write results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = OUTPUT_DIR / f"{timestamp}_test_5min_spanish_e2e_results.log"
    results.write_log(log_file)

    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {len(results.passed)}")
    print(f"Failed: {len(results.failed)}")
    print(f"Translations: {len(results.translations)}")
    print(f"Log file: {log_file}")
    print("=" * 70)

    return len(results.failed) == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
