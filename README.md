<h1 align="center">Plant Disease Prediction</h1>

<p align="center">
  AI-powered crop disease detection and treatment guidance for tomato, potato, and eggplant.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Backend-234f1e?style=for-the-badge" alt="Python Backend" />
  <img src="https://img.shields.io/badge/FastAPI-API-1f6f54?style=for-the-badge" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-Frontend-2b6cb0?style=for-the-badge" alt="React Frontend" />
  <img src="https://img.shields.io/badge/TensorFlow-Classifier-c05621?style=for-the-badge" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/YOLOv8-Detection-6b46c1?style=for-the-badge" alt="YOLOv8" />
</p>

> A full-stack deep learning system that detects affected plant parts, predicts disease classes, and generates treatment guidance through a production-style web app.

## Demo

<p align="center">
  <a href="docs/demo/proj.mp4">Watch / Download the project demo</a>
</p>

> GitHub may show this as a downloadable video link depending on the viewer. Open `docs/demo/proj.mp4` for the full demo.

## Why This Project Stands Out

| Metric | Value |
|---|---:|
| Supported crops | **3** |
| Unified dataset size | **8,158 images** |
| Labeled objects | **12,913** |
| Detection / disease classes | **6** |
| Core models | **2** |
| Split strategy | **80 / 20** |

## Core Features

- YOLOv8-based plant-part localization for more targeted disease analysis
- TensorFlow-based classification pipeline for disease prediction and severity-oriented diagnosis
- React + FastAPI application for upload, inference, visualization, and treatment guidance
- Firebase-backed authentication and history flow for a more complete product experience
- Organized `scripts/` and `docs/` structure for training, inference, verification, and reporting

## Tech Stack

`Python` `FastAPI` `React` `TensorFlow` `YOLOv8` `Firebase`

## Project Layout

```text
website/    full-stack application
scripts/    training, inference, verification, and dataset utilities
docs/       guides, reports, and project notes
```

## Quick Start

### Backend

```bash
cp website/backend/.env.example website/backend/.env
source venv/bin/activate
uvicorn website.backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd website/frontend
cp .env.example .env
npm install
npm run dev
```

## Where To Look

- App code: `website/backend` and `website/frontend`
- Training scripts: `scripts/training`
- Inference scripts: `scripts/inference`
- Verification scripts: `scripts/verification`
- Documentation: `docs/guides`

## Setup Notes

- Keep real secrets only in local `.env` files
- Use `.env.example` files as safe setup templates
- The current detector checkpoint is a YOLO detection model; segmentation support can be added later by swapping model weights
