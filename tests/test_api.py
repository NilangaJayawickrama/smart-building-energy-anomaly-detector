from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


TEST_DATABASE_URL = "sqlite://"


engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={
            "indoor_temperature": 22,
            "outdoor_temperature": 10,
            "humidity": 45,
            "occupancy": 30,
            "hvac_power_kw": 12,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "id" in data
    assert "status" in data
    assert "anomaly_score" in data

    assert data["status"] in [
        "normal",
        "anomaly",
    ]


def test_invalid_humidity():
    response = client.post(
        "/predict",
        json={
            "indoor_temperature": 22,
            "outdoor_temperature": 10,
            "humidity": 150,
            "occupancy": 30,
            "hvac_power_kw": 12,
        },
    )

    assert response.status_code == 422


def test_invalid_occupancy():
    response = client.post(
        "/predict",
        json={
            "indoor_temperature": 22,
            "outdoor_temperature": 10,
            "humidity": 45,
            "occupancy": -5,
            "hvac_power_kw": 12,
        },
    )

    assert response.status_code == 422


def test_prediction_history():
    response = client.get("/predictions")

    assert response.status_code == 200
    assert isinstance(
        response.json(),
        list,
    )


def test_prediction_not_found():
    response = client.get(
        "/predictions/999999"
    )

    assert response.status_code == 404


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Isolation Forest"
    assert "features" in data