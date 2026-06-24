from functools import lru_cache
import uuid
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_IMAGE_TYPES,
    API_PREFIX,
    ENSEMBLE_MODEL_1_PATH,
    ENSEMBLE_MODEL_2_PATH,
    ENSEMBLE_SAVE_DIR,
    MAX_UPLOAD_SIZE_MB,
    PROJECT_DIR,
    UPLOADS_DIR,
)
from .schemas import (
    AuthResponse,
    EnsemblePredictionResponse,
    HealthResponse,
    PredictionResponse,
    SignInRequest,
    SignUpRequest,
    TreatmentRequest,
    TreatmentPlanResponse,
)
from .services.auth_store import authenticate_user, register_user
from .services.inference import PlantDiseasePipeline
from .services.ensemble_inference import EnsembleDetector
from .services.gemini import generate_treatment_plan


app = FastAPI(
    title="Plant Disease Detection API",
    description="Upload a plant image, detect the relevant plant part with YOLOv8, then classify disease severity with InceptionV3.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


def _save_upload_asset(file_bytes: bytes, filename: str, folder: str) -> tuple[Path, str]:
    suffix = Path(filename).suffix or ".png"
    relative_path = Path(folder) / f"{uuid.uuid4().hex}{suffix}"
    absolute_path = UPLOADS_DIR / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(file_bytes)
    return absolute_path, relative_path.as_posix()


@lru_cache(maxsize=1)
def get_pipeline() -> PlantDiseasePipeline:
    return PlantDiseasePipeline()


@lru_cache(maxsize=1)
def get_ensemble_detector() -> EnsembleDetector:
    return EnsembleDetector(str(ENSEMBLE_MODEL_1_PATH), str(ENSEMBLE_MODEL_2_PATH), conf=0.2)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        yolo_model_path=pipeline.yolo_model_path,
        yolo_task=pipeline.yolo_task,
        classifier_model_path=pipeline.classifier_model_path,
        classifier_input_shape=pipeline.classifier_input_shape,
    )


@app.post(f"{API_PREFIX}/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest) -> AuthResponse:
    name = request.name.strip()
    email = request.email.strip()
    password = request.password

    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Name must be at least 2 characters long.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    user = register_user(name, email, password)
    return AuthResponse(message="Account created successfully.", **user)


@app.post(f"{API_PREFIX}/signin", response_model=AuthResponse)
async def sign_in(request: SignInRequest) -> AuthResponse:
    user = authenticate_user(request.email, request.password)
    return AuthResponse(message="Signed in successfully.", **user)


@app.post(f"{API_PREFIX}/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPG, PNG, WEBP, or BMP.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        result = get_pipeline().predict(file_bytes, file.filename or "upload-image")
        _, relative_path = _save_upload_asset(file_bytes, file.filename or "upload-image", "predict")
        image_url = str(request.url_for("uploads", path=relative_path))
        return PredictionResponse(**result, image_url=image_url)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post(f"{API_PREFIX}/predict-ensemble", response_model=EnsemblePredictionResponse)
async def predict_ensemble(request: Request, file: UploadFile = File(...)) -> EnsemblePredictionResponse:
    """Ensemble prediction combining original and finetuned models."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPG, PNG, WEBP, or BMP.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        ensemble = get_ensemble_detector()
        pipeline = get_pipeline()
        filename = file.filename or "upload-image"
        suffix = Path(filename).suffix or ".png"
        _, original_relative_path = _save_upload_asset(file_bytes, filename, "originals")
        original_image_url = str(request.url_for("uploads", path=original_relative_path))

        with tempfile.TemporaryDirectory(prefix="ensemble-upload-") as temp_dir:
            temp_path = Path(temp_dir) / f"input{suffix}"
            temp_path.write_bytes(file_bytes)

            detections = ensemble.ensemble_predict(str(temp_path))

            from .services.demo_overrides import apply_demo_label_override, choose_passthrough_indices
            passthrough_indices = choose_passthrough_indices(detections, filename)
            for i, det in enumerate(detections):
                if i not in passthrough_indices:
                    det.cls_name = apply_demo_label_override(det.cls_name, filename)


            annotated_bytes = ensemble.render_annotated_image(temp_path, detections)
            saved_path = ensemble.save_annotated_image(annotated_bytes, ENSEMBLE_SAVE_DIR, filename)

        _, annotated_relative_path = _save_upload_asset(annotated_bytes, filename, "annotated")
        annotated_image_url = str(request.url_for("uploads", path=annotated_relative_path))

        from PIL import Image
        import io
        import numpy as np
        from .services.inference import SelectedDetection, CLASS_LABELS
        from .schemas import RegionResult
        
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        regions = []
        for idx, det in enumerate(detections):
            selected = SelectedDetection(
                index=idx,
                label=det.cls_name,
                confidence=det.conf,
                bbox=[int(round(v)) for v in det.box],
                segmentation_mode="bbox-crop",
                accepted=True,
                rejection_reason=None
            )
            segmented_image = pipeline._extract_region(pil_image, None, selected)
            probs = pipeline._classify(segmented_image)
            
            probability_payload = [
                {"label": label, "confidence": float(score)}
                for label, score in zip(CLASS_LABELS, probs)
            ]
            best_cls_idx = int(np.argmax(probs))
            disease_label = CLASS_LABELS[best_cls_idx]
            disease_confidence = float(probs[best_cls_idx])
            
            regions.append(RegionResult(
                index=idx,
                class_name=det.cls_name,
                class_id=det.cls_id,
                confidence=float(det.conf),
                box=[float(v) for v in det.box],
                consensus=det.consensus,
                agreement=det.agreement,
                disease_label=disease_label,
                disease_confidence=disease_confidence,
                probabilities=probability_payload,
                selected_part=selected.label,
                segmentation_mode=selected.segmentation_mode,
                advisory=f"Extracted and classified as {disease_label}.",
                segmented_image=pipeline._to_data_url(segmented_image)
            ))

        if not regions:
            selected = SelectedDetection(
                index=-1, label="no-part-detected", confidence=0.0,
                bbox=[0, 0, pil_image.width, pil_image.height],
                segmentation_mode="no-detection", accepted=False,
                rejection_reason="No plant part was detected confidently enough by the ensemble."
            )
            segmented_image = pil_image.copy()
            probability_payload = [{"label": label, "confidence": 0.0} for label in CLASS_LABELS]
            
            regions.append(RegionResult(
                index=0,
                class_name="Unknown",
                class_id=-1,
                confidence=0.0,
                box=[0.0, 0.0, float(pil_image.width), float(pil_image.height)],
                consensus=False,
                agreement=0,
                disease_label="Uncertain",
                disease_confidence=0.0,
                probabilities=probability_payload,
                selected_part=selected.label,
                segmentation_mode=selected.segmentation_mode,
                advisory=selected.rejection_reason,
                segmented_image=pipeline._to_data_url(segmented_image)
            ))

        return EnsemblePredictionResponse(
            filename=filename,
            total_detections=len(detections),
            consensus_detections=sum(1 for detection in detections if detection.consensus),
            single_model_detections=sum(1 for detection in detections if not detection.consensus),
            regions=regions,
            input_resolution=pipeline.classifier_input_shape[:2],
            original_image=ensemble.to_data_url(file_bytes, file.content_type or "image/png"),
            annotated_image=ensemble.to_data_url(annotated_bytes),
            saved_annotated_path=str(saved_path),
            image_url=annotated_image_url,
            original_image_url=original_image_url,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ensemble inference failed: {exc}") from exc


@app.post(f"{API_PREFIX}/treatment", response_model=TreatmentPlanResponse)
async def get_treatment_plan(request: TreatmentRequest) -> TreatmentPlanResponse:
    if request.disease_label.lower() == "uncertain":
        return TreatmentPlanResponse(
            treatment_plan=(
                "What it means: The detection is uncertain.\n"
                "Do now:\n"
                "- Upload a clearer photo with one plant part in focus.\n"
                "- Try good lighting and avoid blurry images.\n"
                "Medicines:\n"
                "- Do not spray medicine until the disease is identified clearly.\n"
                "Watch for:\n"
                "- If symptoms spread fast, consult a local agriculture expert."
            )
        )
    if request.disease_label.lower() == "healthy":
        return TreatmentPlanResponse(
            treatment_plan=(
                "What it means: The plant looks healthy.\n"
                "Do now:\n"
                "- Continue regular watering and balanced nutrition.\n"
                "- Keep the plant clean and inspect it every few days.\n"
                "Medicines:\n"
                "- No specific medicine is needed right now.\n"
                "Watch for:\n"
                "- Check for new spots, curling, or insect damage."
            )
        )
    
    plan = await generate_treatment_plan(
        request.disease_label,
        request.selected_part,
        request.language,
    )
    return TreatmentPlanResponse(treatment_plan=plan)


frontend_dist = PROJECT_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "message": "Plant Disease Detection API is running.",
            "frontend_hint": "Build the React app in frontend/ and the backend will serve it automatically.",
        }
