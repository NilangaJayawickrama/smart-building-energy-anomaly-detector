from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.ml_service import AnomalyDetector
from app.models import Prediction
from app.schemas import SensorReading

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Building Energy Anomaly Detector",
    version="1.0.0",
)

STATIC_DIR = Path("static")

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

detector = AnomalyDetector()

@app.get("/")
def home():
    return FileResponse(
        STATIC_DIR / "index.html"
    )
    

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "building-anomaly-detector",
    }


@app.post("/predict")
def predict(
    reading: SensorReading,
    db: Session = Depends(get_db),
):
    result = detector.predict(
        reading.indoor_temperature,
        reading.outdoor_temperature,
        reading.humidity,
        reading.occupancy,
        reading.hvac_power_kw,
    )

    prediction = Prediction(
        indoor_temperature=reading.indoor_temperature,
        outdoor_temperature=reading.outdoor_temperature,
        humidity=reading.humidity,
        occupancy=reading.occupancy,
        hvac_power_kw=reading.hvac_power_kw,
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return {
        "id": prediction.id,
        "status": "anomaly" if prediction.is_anomaly else "normal",
        "anomaly_score": prediction.anomaly_score,
        "created_at": prediction.created_at,
    }


@app.get("/predictions")
def get_predictions(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    statement = (
        select(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
    )

    return db.scalars(statement).all()


@app.get("/predictions/{prediction_id}")
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    prediction = db.get(
        Prediction,
        prediction_id,
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return prediction


@app.get("/model-info")
def model_info():
    return {
        "model": "Isolation Forest",
        "task": "HVAC energy anomaly detection",
        "features": [
            "indoor_temperature",
            "outdoor_temperature",
            "humidity",
            "occupancy",
            "hvac_power_kw",
        ],
    }