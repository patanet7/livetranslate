import os

import uvicorn


def main() -> None:
    service = os.getenv("MOCK_SERVICE", "whisper").lower()
    default_port = 5003 if service == "translation" else 5001
    port = int(os.getenv("MOCK_PORT", default_port))
    uvicorn.run(
        "scripts.mocks.service:create_app",
        host="0.0.0.0",
        port=port,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    main()
