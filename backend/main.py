import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Dashboard API",
    version="2.1.0",
    description="Robust backend for Sentiment AI Dashboard"
)

# Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
DATA_DIR = Path(__file__).parent / "data" / "exports"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Metric(BaseModel):
    Accuracy: float
    Precision: float
    Recall: float
    F1_Score: float = Field(alias="F1-Score")
    AUC_ROC: float = Field(alias="AUC-ROC")
    Inference_ms: float



class ErrorSample(BaseModel):
    id: int
    model: str
    review: str
    true_label: str
    pred_label: str
    confidence: float
    length: int
    error_type: str

class LimeExample(BaseModel):
    id: int
    model: str
    text: str
    true_label: str
    pred_label: str
    correct: bool
    confidence: float
    words: List[Dict[str, Any]]

class DatasetStatsResponse(BaseModel):
    total_reviews: int
    train_size: int
    test_size: int
    length_distribution: List[Dict[str, Any]]
    length_stats: Dict[str, Any]
    sentiment_distribution: List[Dict[str, Any]]
    top_positive_words: List[Dict[str, Any]]
    top_negative_words: List[Dict[str, Any]]
    length_vs_accuracy: List[Dict[str, Any]]
    sample_reviews: List[Dict[str, Any]]

class ModelAgreementResponse(BaseModel):
    matrix: Dict[str, Dict[str, float]]
    all_wrong: List[Dict[str, Any]]

def load_json_file(filename: str) -> Any:
    """Safely load a JSON file from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise HTTPException(
            status_code=404, 
            detail=f"Data file '{filename}' not found. Please ensure the training step is complete."
        )
    
    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: Failed to parse data file '{filename}'."
        )
    except Exception as e:
        logger.error(f"Unexpected error reading {path}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/", tags=["system"])
def root():
    files = [f.name for f in DATA_DIR.glob("*.json")] if DATA_DIR.exists() else []
    return {
        "status": "ok",
        "version": app.version,
        "mode": "static",
        "available_exports": files
    }

@app.get("/metrics", response_model=Dict[str, Dict[str, float]], tags=["data"])
def get_metrics():
    return load_json_file("metrics.json")

@app.get("/confusion-matrices", tags=["data"])
def get_confusion_matrices():
    return load_json_file("confusion_matrices.json")

@app.get("/roc-curves", tags=["data"])
def get_roc_curves():
    return load_json_file("roc_curves.json")

@app.get("/pr-curves", tags=["data"])
def get_pr_curves():
    return load_json_file("pr_curves.json")

@app.get("/training-history", tags=["data"])
def get_training_history():
    return load_json_file("training_history.json")

@app.get("/errors", response_model=List[ErrorSample], tags=["data"])
def get_errors(model: Optional[str] = None, error_type: Optional[str] = None, min_confidence: float = 0.0):
    data = load_json_file("error_samples.json")
    result = []
    for model_name, errors in data.items():
        for e in errors:
            if model and e.get("model", model_name) != model:
                continue
            if error_type and e.get("error_type", "") != error_type:
                continue
            if e.get("confidence", 0) < min_confidence:
                continue
            e["model"] = e.get("model", model_name)
            result.append(e)
    return result

@app.get("/lime-examples", response_model=List[LimeExample], tags=["data"])
def get_lime_examples(model: Optional[str] = None):
    data = load_json_file("lime_examples.json")
    if model:
        return [e for e in data if e.get("model") == model]
    return data

@app.get("/confidence-distribution", tags=["data"])
def get_confidence_distribution():
    return load_json_file("confidence_dist.json")

@app.get("/feature-importance", tags=["data"])
def get_feature_importance(model: Optional[str] = None):
    data = load_json_file("feature_importance.json")
    if model and model in data:
        return {model: data[model]}
    return data

@app.get("/dataset/stats", tags=["data"])
def get_dataset_stats():
    return load_json_file("dataset_stats.json")

@app.get("/model-agreement", tags=["data"])
def get_model_agreement():
    return load_json_file("model_agreement.json")
