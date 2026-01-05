# Deployment Architecture

## Local Development
All services on localhost (ports 3000, 5001, 5003, 5173, 5432)

## Docker Compose
```yaml
services:
  frontend, orchestration, whisper, translation, postgres
```

## Kubernetes (Production)
- Multiple replicas
- Load balancing
- Auto-scaling
- GPU/NPU node affinity

See [Deployment Scenarios](../01-context/deployment-scenarios.md) for details.
