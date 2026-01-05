# LiveTranslate Documentation

Documentation organized using the **C4 Model** for clear architectural communication from system context to implementation details.

## 📚 Documentation Structure

### 🌍 [Level 1: System Context](./01-context/)
**External view - Who uses the system? How does it fit in the world?**
- [System Overview](./01-context/README.md)
- [Users & Personas](./01-context/users-and-personas.md)
- [Use Cases](./01-context/use-cases.md)
- [External Systems](./01-context/external-systems.md)
- [Deployment Scenarios](./01-context/deployment-scenarios.md)

### 📦 [Level 2: Container Architecture](./02-containers/)
**Service-level view - What are the high-level technical building blocks?**
- [Container Overview](./02-containers/README.md)
- [Service Overview](./02-containers/service-overview.md)
- [Communication Patterns](./02-containers/communication-patterns.md)
- [Data Flow](./02-containers/data-flow.md)
- [Deployment Architecture](./02-containers/deployment-architecture.md)
- [Hardware Optimization](./02-containers/hardware-optimization.md)

### 🔧 [Level 3: Component Details](./03-components/)
**Component-level view - What components exist inside each container?**
- [Component Index](./03-components/README.md)
- [Orchestration Components](./03-components/orchestration/)
- [Whisper Components](./03-components/whisper/)
- [Translation Components](./03-components/translation/)
- [Frontend Components](./03-components/frontend/)

---

## 📖 User & Developer Guides

### Quick Start
- [**5-Minute Setup**](./guides/quick-start.md) - Get LiveTranslate running locally
- [Database Setup](./guides/database-setup.md) - Initialize PostgreSQL
- [Translation Testing](./guides/translation-testing.md) - Test the translation pipeline
- [Deployment Guide](./guides/deployment.md) - Production deployment

### Development
- [Contributing Guide](./guides/contributing.md) - How to contribute
- [Troubleshooting](./guides/troubleshooting.md) - Common issues and solutions

---

## 🔌 API Reference

- [API Index](./api/README.md)
- [Orchestration API](./api/orchestration-api.md)
- [Whisper API](./api/whisper-api.md)
- [Translation API](./api/translation-api.md)
- [WebSocket API](./api/websocket-api.md)

---

## 🔧 Operations & DevOps

- [Monitoring](./operations/monitoring.md) - Metrics and monitoring
- [Logging](./operations/logging.md) - Log aggregation
- [Backup & Restore](./operations/backup-restore.md) - Backup strategies
- [Performance Tuning](./operations/performance-tuning.md) - Optimization guide

---

## 📦 Archive

Historical documentation preserved for reference:
- [Analysis Documents](./archive/analysis/)
- [Planning Documents](./archive/planning/)
- [Status Summaries](./archive/summaries/)

---

## 🚀 Quick Navigation

**New to LiveTranslate?**
1. Start with [System Context](./01-context/README.md)
2. Read [Quick Start Guide](./guides/quick-start.md)
3. Explore [Service Overview](./02-containers/service-overview.md)

**Developer?**
1. Read [Contributing Guide](./guides/contributing.md)
2. Review [Component Documentation](./03-components/README.md)
3. Check [API Reference](./api/README.md)

**DevOps/SRE?**
1. Review [Deployment Architecture](./02-containers/deployment-architecture.md)
2. Read [Operations Guide](./operations/monitoring.md)
3. Check [Deployment Guide](./guides/deployment.md)

---

## 📝 C4 Model

This documentation follows the [C4 model](https://c4model.com/) for visualizing software architecture:

- **Level 1 - Context**: System context and external dependencies
- **Level 2 - Containers**: High-level technology choices (services, databases)
- **Level 3 - Components**: Components within each container

We stop at Level 3 - code implementation details are documented directly in source code comments and docstrings.
