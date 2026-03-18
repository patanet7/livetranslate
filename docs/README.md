# LiveTranslate Documentation Hub

This is the canonical index for active project documentation.

## Start Here

- [Repository README](../README.md)
- [Quick Start Guide](./guides/quick-start.md)
- [Database Setup Guide](./guides/database-setup.md)
- [Translation Testing Guide](./guides/translation-testing.md)
- [Documentation Maintenance](./MAINTENANCE.md)

## Architecture (C4)

- [Level 1 - Context](./01-context/README.md)
- [Level 2 - Containers](./02-containers/README.md)
- [Level 3 - Components](./03-components/README.md)
- [Bot System](./03-components/bot-system.md)

## Service Documentation

- [Modules Index](../modules/README.md)
- [Orchestration Service](../modules/orchestration-service/README.md)
- [Transcription Service](../modules/transcription-service/README.md)
- [Dashboard Service](../modules/dashboard-service/README.md)
- [Shared Module](../modules/shared/README.md)

### Archived Services

- [Whisper Service](../modules/whisper-service/README.md) (replaced by Transcription Service)
- [Translation Service](../modules/translation-service/README.md) (translation now integrated into Orchestration Service via LLM API)
- [Frontend Service](../modules/frontend-service/README.md) (replaced by Dashboard Service)

## Operations, Quality, and Decisions

- [Debugging Guide](./debugging.md)
- [Test Suite Guide](../tests/README.md)
- [ADR 0001 - Microservices Architecture](./adr/0001-microservices-architecture.md)
- [ADR 0002 - PDM Dependency Management](./adr/0002-pdm-dependency-management.md)

## ML and Audit Material

- [ML Pipeline Summary](./ML_PIPELINE_SUMMARY.md)
- [ML Deployment Infrastructure](./ML_DEPLOYMENT_INFRASTRUCTURE.md)
- [DX Findings](./DX_FINDINGS.md)
- [Audit Findings](./audit_findings/)

## Historical Material

- [Archive Index](./archive/README.md)
- [Root-Level Report Index](./archive/root-level-reports.md)
- [Cross-Service Contract Audit](./archive/root-reports/analysis-audit/CROSS_SERVICE_CONTRACT_AUDIT.md)
- [Runtime Surface Archive](../archive/runtime-surfaces/README.md)
