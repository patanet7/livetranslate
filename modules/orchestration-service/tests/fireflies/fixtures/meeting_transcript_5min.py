"""
5-Minute Technical Meeting Transcript Fixture

A realistic multi-speaker technical meeting with:
- 3 speakers (John, Sarah, Mike)
- ~60 sentences over 5 minutes (300 seconds)
- Glossary terms embedded for verification
- Known English text with expected Spanish translations

Used for E2E testing of:
- Context window building
- Glossary term application
- Caption display flow
- Speaker attribution
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TranscriptEntry:
    """Single transcript entry."""
    speaker: str
    text: str
    timestamp_ms: int
    expected_translation: str


# Glossary terms to verify in translations
GLOSSARY_TERMS: Dict[str, str] = {
    "API": "API",
    "endpoint": "punto de acceso",
    "microservice": "microservicio",
    "deployment": "despliegue",
    "rollback": "reversión",
    "database": "base de datos",
    "server": "servidor",
    "client": "cliente",
    "authentication": "autenticación",
    "cache": "caché",
}

# 5-minute meeting transcript (~60 sentences, 5 seconds apart average)
# Format: (speaker, english_text, timestamp_ms, expected_spanish)
MEETING_TRANSCRIPT: List[TranscriptEntry] = [
    # Opening (0-30 seconds)
    TranscriptEntry(
        "John",
        "Good morning everyone. Let's discuss the API changes.",
        0,
        "Buenos días a todos. Vamos a discutir los cambios de la API."
    ),
    TranscriptEntry(
        "Sarah",
        "Thanks John. I've reviewed the endpoint documentation.",
        5000,
        "Gracias John. He revisado la documentación del punto de acceso."
    ),
    TranscriptEntry(
        "Mike",
        "I have some concerns about the deployment process.",
        10000,
        "Tengo algunas preocupaciones sobre el proceso de despliegue."
    ),
    TranscriptEntry(
        "John",
        "Let's hear them. The microservice architecture needs careful planning.",
        15000,
        "Escuchemos. La arquitectura de microservicio necesita una planificación cuidadosa."
    ),
    TranscriptEntry(
        "Mike",
        "We should definitely plan for rollback scenarios.",
        20000,
        "Definitivamente deberíamos planificar escenarios de reversión."
    ),
    TranscriptEntry(
        "Sarah",
        "Agreed. What about the database migrations?",
        25000,
        "De acuerdo. ¿Qué hay de las migraciones de base de datos?"
    ),

    # Technical discussion (30-90 seconds)
    TranscriptEntry(
        "John",
        "The server needs to handle the new authentication flow.",
        30000,
        "El servidor necesita manejar el nuevo flujo de autenticación."
    ),
    TranscriptEntry(
        "Mike",
        "I'll configure the cache settings for better performance.",
        35000,
        "Configuraré los ajustes de caché para mejor rendimiento."
    ),
    TranscriptEntry(
        "Sarah",
        "The client team is waiting for the API specifications.",
        40000,
        "El equipo de cliente está esperando las especificaciones de la API."
    ),
    TranscriptEntry(
        "John",
        "We need to test each endpoint thoroughly.",
        45000,
        "Necesitamos probar cada punto de acceso exhaustivamente."
    ),
    TranscriptEntry(
        "Mike",
        "The deployment pipeline is almost ready.",
        50000,
        "El pipeline de despliegue está casi listo."
    ),
    TranscriptEntry(
        "Sarah",
        "When can we schedule the next microservice review?",
        55000,
        "¿Cuándo podemos programar la próxima revisión de microservicio?"
    ),
    TranscriptEntry(
        "John",
        "Let's plan for Friday. We need the database team.",
        60000,
        "Planeemos para el viernes. Necesitamos al equipo de base de datos."
    ),
    TranscriptEntry(
        "Mike",
        "I'll prepare the server metrics by then.",
        65000,
        "Prepararé las métricas del servidor para entonces."
    ),
    TranscriptEntry(
        "Sarah",
        "The client requirements have changed slightly.",
        70000,
        "Los requisitos del cliente han cambiado ligeramente."
    ),
    TranscriptEntry(
        "John",
        "How does that affect our API design?",
        75000,
        "¿Cómo afecta eso a nuestro diseño de API?"
    ),
    TranscriptEntry(
        "Mike",
        "We might need to update the authentication module.",
        80000,
        "Podríamos necesitar actualizar el módulo de autenticación."
    ),
    TranscriptEntry(
        "Sarah",
        "The cache strategy should remain the same.",
        85000,
        "La estrategia de caché debería permanecer igual."
    ),

    # Deep dive (90-150 seconds)
    TranscriptEntry(
        "John",
        "Let's review the endpoint security requirements.",
        90000,
        "Revisemos los requisitos de seguridad del punto de acceso."
    ),
    TranscriptEntry(
        "Mike",
        "The server firewall rules are configured correctly.",
        95000,
        "Las reglas del firewall del servidor están configuradas correctamente."
    ),
    TranscriptEntry(
        "Sarah",
        "Our client partners need early access to test.",
        100000,
        "Nuestros socios clientes necesitan acceso temprano para probar."
    ),
    TranscriptEntry(
        "John",
        "We can provide a staging API environment.",
        105000,
        "Podemos proporcionar un entorno de API de pruebas."
    ),
    TranscriptEntry(
        "Mike",
        "I'll handle the deployment to staging.",
        110000,
        "Me encargaré del despliegue a pruebas."
    ),
    TranscriptEntry(
        "Sarah",
        "What's the rollback plan if something goes wrong?",
        115000,
        "¿Cuál es el plan de reversión si algo sale mal?"
    ),
    TranscriptEntry(
        "John",
        "The microservice can be rolled back independently.",
        120000,
        "El microservicio puede revertirse independientemente."
    ),
    TranscriptEntry(
        "Mike",
        "The database schema supports backward compatibility.",
        125000,
        "El esquema de base de datos soporta compatibilidad hacia atrás."
    ),
    TranscriptEntry(
        "Sarah",
        "Perfect. The client team will appreciate that.",
        130000,
        "Perfecto. El equipo de cliente lo apreciará."
    ),
    TranscriptEntry(
        "John",
        "Let's document the authentication process clearly.",
        135000,
        "Documentemos el proceso de autenticación claramente."
    ),
    TranscriptEntry(
        "Mike",
        "I'll update the server configuration guide.",
        140000,
        "Actualizaré la guía de configuración del servidor."
    ),
    TranscriptEntry(
        "Sarah",
        "The cache documentation needs updating too.",
        145000,
        "La documentación de caché también necesita actualización."
    ),

    # Planning (150-210 seconds)
    TranscriptEntry(
        "John",
        "The API versioning strategy needs discussion.",
        150000,
        "La estrategia de versionado de API necesita discusión."
    ),
    TranscriptEntry(
        "Mike",
        "We should use semantic versioning for the endpoints.",
        155000,
        "Deberíamos usar versionado semántico para los puntos de acceso."
    ),
    TranscriptEntry(
        "Sarah",
        "The client SDK will need updates accordingly.",
        160000,
        "El SDK de cliente necesitará actualizaciones acorde."
    ),
    TranscriptEntry(
        "John",
        "Let's plan the deployment in phases.",
        165000,
        "Planifiquemos el despliegue en fases."
    ),
    TranscriptEntry(
        "Mike",
        "Phase one covers the core microservice changes.",
        170000,
        "La fase uno cubre los cambios principales del microservicio."
    ),
    TranscriptEntry(
        "Sarah",
        "When does the database migration happen?",
        175000,
        "¿Cuándo ocurre la migración de base de datos?"
    ),
    TranscriptEntry(
        "John",
        "The server team handles that in phase two.",
        180000,
        "El equipo de servidor se encarga de eso en la fase dos."
    ),
    TranscriptEntry(
        "Mike",
        "We need thorough rollback testing before production.",
        185000,
        "Necesitamos pruebas exhaustivas de reversión antes de producción."
    ),
    TranscriptEntry(
        "Sarah",
        "The client team will run integration tests.",
        190000,
        "El equipo de cliente ejecutará pruebas de integración."
    ),
    TranscriptEntry(
        "John",
        "Good. Let's finalize the API contract today.",
        195000,
        "Bien. Finalicemos el contrato de API hoy."
    ),
    TranscriptEntry(
        "Mike",
        "The authentication tokens need longer expiry times.",
        200000,
        "Los tokens de autenticación necesitan tiempos de expiración más largos."
    ),
    TranscriptEntry(
        "Sarah",
        "The cache timeout should match that.",
        205000,
        "El tiempo de espera de caché debería coincidir."
    ),

    # Implementation details (210-270 seconds)
    TranscriptEntry(
        "John",
        "Let's review the endpoint rate limits.",
        210000,
        "Revisemos los límites de velocidad del punto de acceso."
    ),
    TranscriptEntry(
        "Mike",
        "The server can handle a thousand requests per second.",
        215000,
        "El servidor puede manejar mil solicitudes por segundo."
    ),
    TranscriptEntry(
        "Sarah",
        "Our client load testing shows similar numbers.",
        220000,
        "Nuestras pruebas de carga de cliente muestran números similares."
    ),
    TranscriptEntry(
        "John",
        "The API gateway will handle overflow gracefully.",
        225000,
        "La puerta de enlace de API manejará el desbordamiento con gracia."
    ),
    TranscriptEntry(
        "Mike",
        "I'll set up the deployment monitoring dashboard.",
        230000,
        "Configuraré el panel de monitoreo de despliegue."
    ),
    TranscriptEntry(
        "Sarah",
        "Can the client receive real-time status updates?",
        235000,
        "¿Puede el cliente recibir actualizaciones de estado en tiempo real?"
    ),
    TranscriptEntry(
        "John",
        "Yes, through the microservice event stream.",
        240000,
        "Sí, a través del flujo de eventos del microservicio."
    ),
    TranscriptEntry(
        "Mike",
        "The database connection pooling is optimized.",
        245000,
        "El pool de conexiones de base de datos está optimizado."
    ),
    TranscriptEntry(
        "Sarah",
        "What about the authentication refresh flow?",
        250000,
        "¿Qué hay del flujo de actualización de autenticación?"
    ),
    TranscriptEntry(
        "John",
        "The server handles that automatically now.",
        255000,
        "El servidor maneja eso automáticamente ahora."
    ),
    TranscriptEntry(
        "Mike",
        "Cache invalidation is synchronized across nodes.",
        260000,
        "La invalidación de caché está sincronizada entre nodos."
    ),
    TranscriptEntry(
        "Sarah",
        "That's a major improvement for the client experience.",
        265000,
        "Esa es una mejora importante para la experiencia del cliente."
    ),

    # Closing (270-300 seconds)
    TranscriptEntry(
        "John",
        "Let's summarize the API changes we discussed.",
        270000,
        "Resumamos los cambios de API que discutimos."
    ),
    TranscriptEntry(
        "Mike",
        "The deployment schedule looks achievable.",
        275000,
        "El cronograma de despliegue parece alcanzable."
    ),
    TranscriptEntry(
        "Sarah",
        "I'll inform the client team about the timeline.",
        280000,
        "Informaré al equipo de cliente sobre el cronograma."
    ),
    TranscriptEntry(
        "John",
        "The microservice documentation should be ready tomorrow.",
        285000,
        "La documentación del microservicio debería estar lista mañana."
    ),
    TranscriptEntry(
        "Mike",
        "I'll complete the server configuration today.",
        290000,
        "Completaré la configuración del servidor hoy."
    ),
    TranscriptEntry(
        "Sarah",
        "Great meeting everyone. Thanks for the thorough discussion.",
        295000,
        "Gran reunión todos. Gracias por la discusión exhaustiva."
    ),
    TranscriptEntry(
        "John",
        "Thank you. Let's reconvene Friday with progress updates.",
        300000,
        "Gracias. Reunámonos el viernes con actualizaciones de progreso."
    ),
]


def get_transcript_as_tuples() -> List[Tuple[str, str, int]]:
    """Get transcript as simple tuples (speaker, text, timestamp_ms)."""
    return [(e.speaker, e.text, e.timestamp_ms) for e in MEETING_TRANSCRIPT]


def get_expected_translations() -> Dict[int, str]:
    """Get mapping of timestamp_ms -> expected translation."""
    return {e.timestamp_ms: e.expected_translation for e in MEETING_TRANSCRIPT}


def get_entries_with_glossary_term(term: str) -> List[TranscriptEntry]:
    """Get entries that contain a specific glossary term."""
    return [e for e in MEETING_TRANSCRIPT if term.lower() in e.text.lower()]


def get_transcript_duration_seconds() -> float:
    """Get total duration of transcript in seconds."""
    if not MEETING_TRANSCRIPT:
        return 0.0
    return MEETING_TRANSCRIPT[-1].timestamp_ms / 1000.0


def get_speaker_sentence_count() -> Dict[str, int]:
    """Get count of sentences per speaker."""
    counts: Dict[str, int] = {}
    for entry in MEETING_TRANSCRIPT:
        counts[entry.speaker] = counts.get(entry.speaker, 0) + 1
    return counts


# Verification helpers
GLOSSARY_VERIFICATION_CASES = [
    # (timestamp_ms, english_term, expected_spanish_term)
    (5000, "endpoint", "punto de acceso"),
    (10000, "deployment", "despliegue"),
    (15000, "microservice", "microservicio"),
    (20000, "rollback", "reversión"),
    (25000, "database", "base de datos"),
    (30000, "server", "servidor"),
    (30000, "authentication", "autenticación"),
    (35000, "cache", "caché"),
    (40000, "client", "cliente"),
    (40000, "API", "API"),
]


# Context window verification - expected context for each sentence
def get_expected_context_count(index: int, window_size: int = 3) -> int:
    """
    Get expected number of context sentences for a given sentence index.

    First sentence has 0 context (cold start).
    Second has 1, third has 2, then capped at window_size.
    """
    return min(index, window_size)
