---
title: MedCheck
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---
# MedCheck — Prescription Error Detector 💊

An OpenEnv environment where an AI agent analyzes patient profiles and detects dangerous prescription errors before they reach the patient.

## The Problem
Medication errors are one of the leading causes of preventable death in healthcare. 
MedCheck trains AI agents to catch these errors automatically.

## What the Agent Must Do
Given a patient profile (allergies, conditions, current medications) and a new prescription,
the agent must identify dangerous errors and classify their severity.

## Tasks

| Task | Scenario | Difficulty |
|------|----------|------------|
| Easy | Patient allergic to Penicillin prescribed Amoxicillin | Easy |
| Medium | Patient on Warfarin prescribed Aspirin (bleeding risk) | Medium |
| Hard | Kidney disease patient prescribed Metformin overdose + Ibuprofen | Hard |

## Scoring
- **1.0** — All errors detected, severity correctly classified
- **0.5** — Some errors detected
- **0.0** — No errors detected

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit agent action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks |

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn pydantic openai

# Run locally
uvicorn main:app --reload

# Run with Docker
docker build -t medcheck .
docker run -p 7860:7860 medcheck
```

## Environment Variables
- `API_BASE_URL` — LLM API endpoint
- `MODEL_NAME` — Model identifier
- `HF_TOKEN` — Hugging Face / API key

## Action Format
```json
{
  "detected_errors": ["penicillin allergy"],
  "severity": "critical",
  "recommendation": "Switch to a non-penicillin antibiotic"
}
```

## Real World Impact
This environment can be used to train and evaluate AI agents for:
- Hospital prescription validation systems
- Pharmacy safety checks
- Clinical decision support tools