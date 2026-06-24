# Plant Disease Prediction

AI-powered crop disease detection and treatment guidance for tomato, potato, and eggplant using computer vision and deep learning.

## Demo

Add your demo video link here.

## Highlights

- Detects and analyzes diseases across **3 crops**
- Built on a unified dataset of **8,158 images** and **12,913 labeled objects**
- Combines **YOLOv8** plant-part detection with **TensorFlow** disease classification
- Delivers treatment guidance through a **React + FastAPI** web app

## Tech Stack

`Python` `FastAPI` `React` `TensorFlow` `YOLOv8` `Firebase`

## Project Structure

```text
website/    # Full-stack application
scripts/    # Training, inference, verification, and dataset utilities
docs/       # Guides and technical notes
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

## Repository Guide

- App code: `website/backend` and `website/frontend`
- Training scripts: `scripts/training`
- Inference scripts: `scripts/inference`
- Verification scripts: `scripts/verification`
- Documentation: `docs/guides`

## Notes

- Keep real secrets only in local `.env` files
- `.env.example` files are safe templates for setup
- The current detector checkpoint is a YOLO detection model, with segmentation support possible if you later switch weights
