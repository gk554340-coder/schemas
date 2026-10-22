# Autonomous AI Cinema & Documentary Studio Engine
## Master Architecture Blueprint — Version 1.0.0

**Document:** ARCHITECTURE.md  
**Status:** Master Architecture Specification  
**Version:** 1.0.0  
**Python:** 3.11  
**Execution Model:** Modular, Isolated, Checkpoint-Based Pipeline  
**API Framework:** FastAPI + WebSockets  
**Rendering Engine:** FFmpeg  
**CI/CD:** GitHub Actions  
**Test Environment:** CPU-only, GPU-independent test mode

---

# 1. PROJECT VISION

The Autonomous AI Cinema & Documentary Studio Engine is a modular production system designed to transform a user-provided topic into a completed video through five independent production slots.

The system must support research, screenplay creation, audio generation, visual production, and final editing without coupling all functionality into a single monolithic module.

The architecture must prioritize:

1. Modular isolation.
2. Predictable data contracts.
3. Independent stage execution.
4. Checkpoint-based recovery.
5. SHA-256 artifact integrity verification.
6. GPU-independent testing.
7. Structured API integration.
8. Extensibility without redesigning the entire engine.
9. Safe file access and controlled output directories.
10. Observable execution status and error reporting.

This document is the intended architectural source of truth for the repository.

---

# 2. CORE ARCHITECTURAL PRINCIPLES

## 2.1 Single Responsibility

Every slot must own one production responsibility.

A slot must not silently perform another slot's primary responsibility.

## 2.2 Strict Pipeline Order

The production sequence is immutable:

1. `slot1_research.py`
2. `slot2_screenplay.py`
3. `slot4_audio.py`
4. `slot3_visuals.py`
5. `slot5_editor.py`

The audio slot intentionally runs before the visual slot.

The master pipeline must not reorder these stages based on directory names or file numbering.

## 2.3 Data-Contract-First Design

Slots must exchange validated structured data rather than depend on undocumented internal variables.

Shared contracts are defined through `schemas.py` and the JSON Schema files.

## 2.4 Isolation

Each slot must operate inside its assigned workspace.

A failure in one stage must not silently corrupt the files belonging to another stage.

## 2.5 Recovery

Completed stages should not be rerun unnecessarily when their outputs remain valid.

Recovery must validate the stored checkpoint, artifact existence, and artifact checksums before reusing an output.

## 2.6 Testability

The test pipeline must not require:

- A CUDA-capable GPU.
- GPU model weights.
- Live research API access.
- Live AI API credentials.
- Paid inference services.

The test pipeline must use deterministic local fixtures and CPU-compatible FFmpeg operations.

## 2.7 Honest Completion Reporting

A stage may only be reported as completed after its required outputs have been validated.

A generated JSON file that merely describes a future video is not equivalent to a successfully rendered video.

---

# 3. REPOSITORY STRUCTURE

The repository must maintain the following layout:

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
│
├── config.py
├── schemas.py
│
├── slot1_research.py
├── slot2_screenplay.py
├── slot4_audio.py
├── slot3_visuals.py
├── slot5_editor.py
│
├── master_pipeline.py
├── server.py
│
├── schemas/
│   ├── research_output.schema.json
│   ├── screenplay_blueprint.schema.json
│   ├── audio_timeline.schema.json
│   └── project_state.schema.json
│
├── tests/
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_master_pipeline.py
│   ├── test_server.py
│   └── services/
│       └── ai/
│           ├── test_provider.py
│           ├── test_tools.py
│           └── test_orchestrator.py
│
└── workspace/
    ├── 01_research_vault/
    ├── 02_screenplay_desk/
    ├── 03_audio_studio/
    ├── 04_visual_lab/
    └── 05_master_theater/
