#!/bin/bash
# Load secrets from files into environment variables
# Source this script in your containers or applications

if [ -f .secrets/jwt_secret.txt ]; then
    JWT_SECRET=$(cat .secrets/jwt_secret.txt)
    export JWT_SECRET
fi

if [ -f .secrets/db_password.txt ]; then
    POSTGRES_PASSWORD=$(cat .secrets/db_password.txt)
    export POSTGRES_PASSWORD
fi

if [ -f .secrets/redis_password.txt ]; then
    REDIS_PASSWORD=$(cat .secrets/redis_password.txt)
    export REDIS_PASSWORD
fi

if [ -f .secrets/api_key_openai.txt ]; then
    OPENAI_API_KEY=$(cat .secrets/api_key_openai.txt)
    export OPENAI_API_KEY
fi

if [ -f .secrets/api_key_anthropic.txt ]; then
    ANTHROPIC_API_KEY=$(cat .secrets/api_key_anthropic.txt)
    export ANTHROPIC_API_KEY
fi

if [ -f .secrets/api_key_google.txt ]; then
    GOOGLE_API_KEY=$(cat .secrets/api_key_google.txt)
    export GOOGLE_API_KEY
fi

if [ -f .secrets/session_secret.txt ]; then
    SESSION_SECRET=$(cat .secrets/session_secret.txt)
    export SESSION_SECRET
fi

echo "Secrets loaded from .secrets/ directory"
