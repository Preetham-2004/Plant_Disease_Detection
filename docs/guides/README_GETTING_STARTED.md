# Getting Started

This repository has two main parts:

- `website/`: the app you can run locally
- `scripts/`: training, inference, verification, and dataset utilities

## Recommended Order

1. Read the root `README.md`
2. Start the backend from `website/backend`
3. Start the frontend from `website/frontend`
4. Use `scripts/` only when you need training, evaluation, or dataset maintenance

## Project Map

```text
website/
├── backend/
│   └── app/
└── frontend/

scripts/
├── training/
├── inference/
├── verification/
├── dataset/
└── monitoring/

docs/
├── guides/
└── reports/
```

## Run The App

### Backend

From the repository root:

```bash
source venv/bin/activate
uvicorn website.backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd website/frontend
npm install
npm run dev
```

## Useful Script Locations

- Training: `scripts/training/`
- Standalone inference: `scripts/inference/`
- Verification and audits: `scripts/verification/`
- Dataset conversion and repair: `scripts/dataset/`

Run utility scripts from the repository root so existing relative paths resolve correctly.

## Recommended Reading

- `docs/reports/IMPLEMENTATION_SUMMARY.md`
- `docs/reports/DATASET_MERGE_REPORT.md`
- `docs/reports/CLASS_MAPPING_VERIFICATION.md`
