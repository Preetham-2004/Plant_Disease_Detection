# Plant Disease Prediction

An end-to-end plant disease detection project with a FastAPI backend, React frontend, YOLO-based plant-part detection, and InceptionV3-based disease classification.

## Repository Structure

```text
Plant Disease prediction/
├── website/                # Production app
│   ├── backend/            # FastAPI API and inference services
│   └── frontend/           # React + Vite client
├── scripts/                # Training, inference, dataset, and verification utilities
├── docs/                   # Guides, reports, and project notes
├── models/                 # Saved classifier checkpoints
├── models_v2/              # Updated classifier checkpoints
├── runs/                   # YOLO training outputs
├── Dataset/                # Source classification datasets
├── YoloDataset/            # Detection datasets and YOLO assets
└── unified_plant_dataset/  # Unified detector dataset
```

## What This Project Includes

- `website/backend`: FastAPI endpoints for prediction, authentication, and treatment-plan generation
- `website/frontend`: React UI for upload, diagnosis history, and result visualization
- `scripts/training`: Model training entry points
- `scripts/inference`: Standalone inference and analysis utilities
- `scripts/verification`: Dataset and pipeline validation scripts
- `scripts/dataset`: Dataset conversion and repair helpers
- `docs/guides`: Start-here documentation
- `docs/reports`: Technical reports and implementation notes

## Quick Start

### Backend

```bash
source venv/bin/activate
uvicorn website.backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Environment setup:

```bash
cp website/backend/.env.example website/backend/.env
```

Useful endpoints:

- `GET /api/health`
- `POST /api/predict`
- `POST /api/predict-ensemble`

### Frontend

```bash
cd website/frontend
npm install
cp .env.example .env
npm run dev
```

If needed, set:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Recommended Entry Points

- Start here: `docs/guides/README_GETTING_STARTED.md`
- Training utilities: `scripts/training/`
- Inference utilities: `scripts/inference/`
- Verification utilities: `scripts/verification/`
- App code: `website/backend/` and `website/frontend/`

## Secrets And Config

- Real secrets must stay only in local `.env` files
- Share setup using `.env.example` files only
- Frontend config template: `website/frontend/.env.example`
- Backend config template: `website/backend/.env.example`

## Notes

- The current saved YOLO checkpoint in this repository is a detection model, not a segmentation model.
- The backend can use segmentation masks if you later replace the detector with a YOLO segmentation checkpoint.
- Generated outputs, local environments, datasets, and checkpoints are intentionally kept out of Git tracking via `.gitignore`.
