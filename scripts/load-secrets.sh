#!/bin/bash
# Load secrets from files into environment variables
# Source this script in your containers or applications

if [ -f .secrets/jwt_secret.txt ]; then
    export JWT_SECRET=$(cat .secrets/jwt_secret.txt)
fi

if [ -f .secrets/db_password.txt ]; then
    export POSTGRES_PASSWORD=$(cat .secrets/db_password.txt)
fi

if [ -f .secrets/redis_password.txt ]; then
    export REDIS_PASSWORD=$(cat .secrets/redis_password.txt)
fi

if [ -f .secrets/api_key_openai.txt ]; then
    export OPENAI_API_KEY=$(cat .secrets/api_key_openai.txt)
fi

if [ -f .secrets/api_key_anthropic.txt ]; then
    export ANTHROPIC_API_KEY=$(cat .secrets/api_key_anthropic.txt)
fi

if [ -f .secrets/api_key_google.txt ]; then
    export GOOGLE_API_KEY=$(cat .secrets/api_key_google.txt)
fi

if [ -f .secrets/session_secret.txt ]; then
    export SESSION_SECRET=$(cat .secrets/session_secret.txt)
fi

echo "Secrets loaded from .secrets/ directory"
