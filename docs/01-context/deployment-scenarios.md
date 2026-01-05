# Deployment Scenarios

## 1. Local Development (Single Machine)

**Use Case**: Developer testing, personal use
**Configuration**: All services on localhost
**Hardware**: Laptop/desktop with 16GB+ RAM

```
Services:
- Frontend: localhost:5173
- Orchestration: localhost:3000
- Whisper: localhost:5001
- Translation: localhost:5003
- PostgreSQL: localhost:5432
```

**Pros**: Simple setup, fast iteration
**Cons**: Limited scalability, single point of failure

---

## 2. Docker Compose (Single Host)

**Use Case**: Small team, staging environment
**Configuration**: All services in Docker
**Hardware**: Single server with GPU/NPU

```bash
docker-compose up
```

**Pros**: Reproducible, easy deployment
**Cons**: Limited to single host capacity

---

## 3. Kubernetes (Production)

**Use Case**: Enterprise deployment, high availability
**Configuration**: Multi-node cluster
**Hardware**: Kubernetes cluster with GPU nodes

```
Namespaces:
- livetranslate-prod
- livetranslate-staging

Services:
- Frontend: LoadBalancer
- Orchestration: ClusterIP (scaled 3+)
- Whisper: NodePort (GPU affinity)
- Translation: NodePort (GPU affinity)
```

**Pros**: Auto-scaling, high availability, rolling updates
**Cons**: Complex setup, higher cost

---

## 4. Hybrid Cloud

**Use Case**: Data sovereignty + cloud scale
**Configuration**: On-prem Whisper, cloud Translation
**Hardware**: On-prem NPU server + cloud GPU

```
On-Premise:
- Whisper Service (NPU)
- PostgreSQL

Cloud:
- Orchestration
- Translation (GPU instances)
- Frontend
```

**Pros**: Data control, cloud scalability
**Cons**: Network latency, complex networking

---

## Hardware Requirements

See [Hardware Optimization](../02-containers/hardware-optimization.md) for NPU/GPU/CPU strategies.
