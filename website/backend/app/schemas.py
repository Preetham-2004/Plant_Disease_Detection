from pydantic import BaseModel


class DetectionResult(BaseModel):
    label: str
    confidence: float
    bbox: list[int]
    selected: bool


class ProbabilityResult(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    filename: str
    disease_label: str
    disease_confidence: float
    probabilities: list[ProbabilityResult]
    detections: list[DetectionResult]
    selected_part: str
    segmentation_mode: str
    input_resolution: list[int]
    advisory: str
    original_image: str
    segmented_image: str
    annotated_image: str
    image_url: str | None = None


class RegionResult(BaseModel):
    index: int
    class_name: str
    class_id: int
    confidence: float
    box: list[float]
    consensus: bool
    agreement: int
    disease_label: str
    disease_confidence: float
    probabilities: list[ProbabilityResult]
    selected_part: str
    segmentation_mode: str
    advisory: str
    segmented_image: str


class EnsemblePredictionResponse(BaseModel):
    filename: str
    total_detections: int
    consensus_detections: int
    single_model_detections: int
    regions: list[RegionResult]
    input_resolution: list[int]
    original_image: str
    annotated_image: str
    saved_annotated_path: str
    image_url: str | None = None
    original_image_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    yolo_model_path: str
    yolo_task: str
    classifier_model_path: str
    classifier_input_shape: list[int]


class TreatmentRequest(BaseModel):
    disease_label: str
    selected_part: str
    language: str = "English"


class TreatmentPlanResponse(BaseModel):
    treatment_plan: str


class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    message: str
    name: str
    email: str
