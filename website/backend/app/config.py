import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_DIR / "website" / "backend"
DATA_DIR = BACKEND_DIR / "data"

# Load env files from stable locations so backend config works regardless of cwd.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env", override=True)

MODAL_MODEL_ROOT = Path(os.getenv("PLANTGUARD_MODEL_ROOT", "")).expanduser() if os.getenv("PLANTGUARD_MODEL_ROOT") else None
MODAL_UPLOADS_ROOT = Path(os.getenv("PLANTGUARD_UPLOADS_DIR", "")).expanduser() if os.getenv("PLANTGUARD_UPLOADS_DIR") else None

if MODAL_MODEL_ROOT:
    YOLO_MODEL_PATH = MODAL_MODEL_ROOT / "yolo" / "best.pt"
    CLASSIFIER_MODEL_PATH = MODAL_MODEL_ROOT / "classifier" / "inceptionv3_severity_best.keras"
    ENSEMBLE_MODEL_1_PATH = MODAL_MODEL_ROOT / "ensemble" / "model1.pt"
    ENSEMBLE_MODEL_2_PATH = MODAL_MODEL_ROOT / "ensemble" / "model2.pt"
else:
    YOLO_MODEL_PATH = PROJECT_DIR / "runs" / "detect" / "train_yolov8s_cleaned" / "weights" / "best.pt"
    CLASSIFIER_MODEL_PATH = PROJECT_DIR / "models_v2" / "inceptionv3_severity_best.keras"
    ENSEMBLE_MODEL_1_PATH = PROJECT_DIR / "runs" / "detect" / "train_yolov8s_unified_finetune" / "weights" / "best.pt"
    ENSEMBLE_MODEL_2_PATH = PROJECT_DIR / "best_yolo_parts.pt"

ENSEMBLE_SAVE_DIR = PROJECT_DIR / "runs" / "detect" / "ensemble_results"
UPLOADS_DIR = MODAL_UPLOADS_ROOT if MODAL_UPLOADS_ROOT else BACKEND_DIR / "uploads"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
API_PREFIX = "/api"
DEFAULT_YOLO_CONFIDENCE = 0.25
MAX_UPLOAD_SIZE_MB = 10
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
USER_STORE_PATH = DATA_DIR / "users.json"
