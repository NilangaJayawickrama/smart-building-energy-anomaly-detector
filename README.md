# Smart Building Energy Anomaly Detector

A Python-based web application that uses machine learning to detect unusual HVAC energy-consumption patterns from building sensor data.

The project combines a FastAPI backend, an Isolation Forest anomaly-detection model, SQL-based persistence, and a lightweight HTML/CSS/JavaScript frontend.

## Project Motivation

Modern building automation systems continuously generate environmental and energy-consumption data such as temperature, humidity, occupancy, and HVAC power usage.

Abnormal energy-consumption patterns may indicate inefficient system operation, unusual environmental conditions, or equipment-related issues.

This project demonstrates how Python and machine learning can be used to identify unusual HVAC energy-consumption patterns through a web-based application.

## Main Features

- Python-based backend using FastAPI
- Machine-learning anomaly detection using Isolation Forest
- RESTful API for submitting building sensor readings
- Input validation using Pydantic
- SQL database persistence using SQLAlchemy
- Browser-based dashboard using HTML, CSS, and JavaScript
- Prediction history
- Automated API testing using pytest
- Interactive API documentation using FastAPI/OpenAPI
- Docker containerization for portable deployment

## Technologies

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning and Data Processing
- scikit-learn
- pandas
- NumPy
- joblib

### Database
- SQLAlchemy
- SQLite

### Frontend
- HTML
- CSS
- JavaScript

### Testing
- pytest
- HTTPX

### Development and Deployment
- Git
- GitHub
- Docker

---

# Python Packages

The project uses the following Python packages.

## FastAPI

**Package:** `fastapi`

FastAPI is the main Python web framework used to build the backend REST API.

It is responsible for:

- Defining API endpoints such as `/predict` and `/predictions`
- Receiving HTTP requests
- Returning JSON responses
- Validating request data
- Generating interactive API documentation
- Connecting the frontend with the Python backend

Example:

```python
@app.post("/predict")
def predict(reading: SensorReading):
    ...
```

FastAPI automatically provides API documentation at:

```text
/docs
```

---

## Uvicorn

**Package:** `uvicorn`

Uvicorn is the application server used to run the FastAPI application.

FastAPI defines the application and its endpoints, while Uvicorn starts the web server and handles incoming HTTP requests.

The application can be started using:

```bash
python -m uvicorn app.main:app --reload
```

The `--reload` option automatically restarts the development server whenever Python source files are changed.

---

## SQLAlchemy

**Package:** `sqlalchemy`

SQLAlchemy is used as the database ORM, or Object Relational Mapper.

Instead of writing raw SQL queries throughout the application, SQLAlchemy allows database tables to be represented using Python classes.

It is used to:

- Define the `Prediction` database table
- Insert prediction records
- Retrieve prediction history
- Manage database sessions
- Keep database logic organized

Example:

```python
class Prediction(Base):
    __tablename__ = "predictions"
```

The project currently uses SQLite, but SQLAlchemy makes it easier to migrate to databases such as PostgreSQL later.

---

## scikit-learn

**Package:** `scikit-learn`

scikit-learn provides the machine-learning functionality used by the project.

The application uses the `IsolationForest` algorithm for anomaly detection.

The model learns typical relationships between:

- Indoor temperature
- Outdoor temperature
- Humidity
- Occupancy
- HVAC power consumption

It then identifies sensor readings that appear significantly different from normal patterns.

Example:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
```

---

## pandas

**Package:** `pandas`

pandas is used for data loading, transformation, and preparation.

It is used to:

- Read the building sensor dataset
- Store sensor readings in DataFrames
- Select machine-learning features
- Prepare individual sensor readings before sending them to the trained model

Example:

```python
data = pd.read_csv("data/building_data.csv")
```

---

## NumPy

**Package:** `numpy`

NumPy provides numerical and array-processing functionality.

In this project it is mainly used to generate synthetic building sensor data.

It helps create realistic distributions for:

- Indoor temperatures
- Outdoor temperatures
- Humidity
- Occupancy
- HVAC energy consumption

Example:

```python
outdoor_temperature = np.random.normal(
    15,
    8,
    NUM_SAMPLES
)
```

NumPy also provides efficient numerical operations used indirectly by pandas and scikit-learn.

---

## joblib

**Package:** `joblib`

joblib is used to save and load the trained machine-learning model.

After training the Isolation Forest model, it is serialized into a file:

```text
models/isolation_forest.pkl
```

Example:

```python
joblib.dump(
    model,
    "models/isolation_forest.pkl"
)
```

The FastAPI application later loads the same trained model:

```python
model = joblib.load(
    "models/isolation_forest.pkl"
)
```

This avoids retraining the model every time the application starts.

---

## pytest

**Package:** `pytest`

pytest is the testing framework used to create automated tests for the backend.

Tests are used to confirm that important application functionality continues to work correctly.

Examples include testing:

- The health endpoint
- Valid prediction requests
- Invalid sensor values
- Prediction history
- API response status codes

Tests can be executed using:

```bash
python -m pytest -v
```

---

## HTTPX

**Package:** `httpx`

HTTPX is an HTTP client library.

In this project, it supports FastAPI's testing tools and allows automated tests to send requests to the application without manually opening a browser.

For example, a test can submit a request to:

```text
POST /predict
```

and verify the response.

Example:

```python
response = client.post(
    "/predict",
    json={
        "indoor_temperature": 22,
        "outdoor_temperature": 10,
        "humidity": 45,
        "occupancy": 30,
        "hvac_power_kw": 12
    }
)
```

---

# System Architecture

The application follows a simple layered architecture.

```text
                  ┌─────────────────────┐
                  │   Web Dashboard     │
                  │ HTML / CSS / JS     │
                  └─────────┬───────────┘
                            │
                            │ HTTP / JSON
                            ▼
                  ┌─────────────────────┐
                  │      FastAPI        │
                  │   Python Backend    │
                  └──────┬────────┬─────┘
                         │        │
                Prediction       Database
                         │        │
                         ▼        ▼
               ┌────────────┐   ┌────────────┐
               │ Isolation  │   │   SQLite   │
               │   Forest   │   │ SQLAlchemy │
               └────────────┘   └────────────┘
```

## Application Flow

1. A user enters building sensor information in the browser.
2. JavaScript sends the sensor information to the FastAPI backend.
3. FastAPI validates the request using Pydantic.
4. The trained Isolation Forest model analyzes the sensor values.
5. The model returns an anomaly classification and anomaly score.
6. The prediction is stored in the SQLite database.
7. FastAPI returns the prediction result to the frontend.
8. JavaScript displays the result to the user.
9. Previous predictions can be retrieved from the database.

---

# Input Features

The machine-learning model uses five input features.

| Feature | Description |
|---|---|
| Indoor Temperature | Current indoor building temperature in °C |
| Outdoor Temperature | Current outdoor temperature in °C |
| Humidity | Indoor relative humidity percentage |
| Occupancy | Number of occupants currently using the building |
| HVAC Power | Current HVAC power consumption in kW |

Example sensor reading:

```json
{
  "indoor_temperature": 22.0,
  "outdoor_temperature": 10.0,
  "humidity": 45,
  "occupancy": 30,
  "hvac_power_kw": 12.0
}
```

---

# Machine-Learning Approach

The application uses an Isolation Forest model from scikit-learn for unsupervised anomaly detection. The model is trained using indoor temperature, outdoor temperature, humidity, occupancy, and HVAC power consumption. A contamination threshold of 5% is used during training to identify unusual combinations of environmental conditions and energy consumption. The trained model is serialized using joblib and loaded by the FastAPI backend for inference.

Isolation Forest is an unsupervised machine-learning algorithm designed to identify unusual data points.

Instead of predicting a predefined class, the algorithm learns patterns in normal data and identifies observations that are easier to isolate from the rest of the dataset.

For this project, unusual combinations such as:

- Low occupancy with unusually high HVAC consumption
- Small indoor/outdoor temperature difference with unusually high energy use
- Energy usage significantly outside normal operating patterns

may be identified as anomalies.

The model returns:

- `normal`
- `anomaly`

along with an anomaly score.

---

# Dataset

The first version of the project uses a synthetic building sensor dataset generated using Python, NumPy, and pandas.

The generated dataset contains approximately 3,000 observations.

Each observation contains:

- Indoor temperature
- Outdoor temperature
- Humidity
- Occupancy
- HVAC power consumption

The synthetic dataset was selected so the complete machine-learning workflow could be demonstrated without relying on an external proprietary building dataset.

A future version could replace the synthetic dataset with real building automation or BACnet sensor data.

---

# Project Structure

```text
smart-building-energy-anomaly-detector/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── ml_service.py
│
├── data/
│   └── building_data.csv
│
├── models/
│   └── isolation_forest.pkl
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── tests/
│   └── test_api.py
│
├── generate_data.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

# Installation

## Prerequisites

The project was developed using:

- Windows
- Python 3.14
- Visual Studio Code
- Git
- Docker Desktop

## Clone the Repository

```bash
git clone <repository-url>

cd smart-building-energy-anomaly-detector
```

## Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

# Generate the Dataset

Run:

```powershell
python generate_data.py
```

This creates:

```text
data/building_data.csv
```

---

# Train the Machine-Learning Model

Run:

```powershell
python train_model.py
```

The trained Isolation Forest model is saved as:

```text
models/isolation_forest.pkl
```

---

# Run the Application

Start the FastAPI server:

```powershell
python -m uvicorn app.main:app --reload
```

Open the web application:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```text
GET /health
```

Returns information about the API status.

Example:

```json
{
  "status": "healthy",
  "service": "building-anomaly-detector"
}
```

## Create Prediction

```text
POST /predict
```

Example request:

```json
{
  "indoor_temperature": 22,
  "outdoor_temperature": 10,
  "humidity": 45,
  "occupancy": 30,
  "hvac_power_kw": 12
}
```

Example response:

```json
{
  "id": 1,
  "status": "normal",
  "anomaly_score": 0.08,
  "created_at": "2026-09-30T17:00:00"
}
```

## Retrieve Prediction History

```text
GET /predictions
```

Returns recently stored predictions.

## Retrieve Individual Prediction

```text
GET /predictions/{id}
```

Returns a single prediction based on its database ID.

## Model Information

```text
GET /model-info
```

Returns information about the current machine-learning model and input features.

---

# Testing

Automated API tests are written using pytest.

Run:

```powershell
python -m pytest -v
```

Tests include:

- Health endpoint validation
- Valid prediction request
- Invalid input validation
- Prediction history retrieval

---

# Docker

The application can also be run inside a Docker container.

## Build the Image

```powershell
docker build -t smart-building-anomaly-detector .
```

## Run the Container

```powershell
docker run -p 8000:8000 smart-building-anomaly-detector
```

Then open:

```text
http://localhost:8000
```

---

# Development Environment

The application was developed natively on Windows using:

- Python virtual environments
- Visual Studio Code
- PowerShell
- Git
- Docker Desktop

The application itself uses platform-independent relative paths so it can also run inside a Linux-based Docker container.

---

# Design Decisions

## Why FastAPI?

FastAPI was selected because it provides:

- Simple REST API development
- Python type-hint integration
- Input validation
- Automatic OpenAPI documentation
- Strong support for modern Python backend development

## Why Isolation Forest?

Isolation Forest was selected because the task involves detecting unusual building energy-consumption patterns without requiring manually labelled anomaly data.

## Why SQLite?

SQLite was selected for the first prototype because it:

- Requires no separate database server
- Is easy to configure
- Works well for a small demonstration application
- Allows the project to remain self-contained

SQLAlchemy is used so the database layer can be migrated to PostgreSQL in a future version.

## Why Separate the ML Service?

Machine-learning logic is separated from API routing so that:

- Code is easier to maintain
- The ML model can be tested independently
- Different machine-learning models can be introduced later
- API logic remains focused on HTTP request processing

---

# Future Improvements

Possible future improvements include:

- Replace synthetic data with real building automation data
- Integrate BACnet sensor information
- Migrate from SQLite to PostgreSQL
- Deploy the application to AWS
- Add real-time building sensor ingestion
- Add time-series charts
- Add email or notification alerts
- Add authentication
- Add model-performance monitoring
- Add model retraining workflows
- Add Docker Compose
- Deploy the database separately
- Introduce microservices where appropriate
- Add CI/CD using GitHub Actions
- Add Nginx as a reverse proxy
- Explore Kubernetes deployment
- Compare Isolation Forest with additional anomaly-detection algorithms

---

# Skills Demonstrated

This project demonstrates practical experience with:

- Python development
- FastAPI
- RESTful API design
- Machine learning
- Anomaly detection
- Data processing
- SQL databases
- ORM development
- JavaScript
- HTML/CSS
- Input validation
- Automated testing
- Git version control
- Docker
- Backend architecture
- Debugging
- Software documentation

---

# Disclaimer

This project is an educational software prototype.

The synthetic data and anomaly-detection results should not be treated as real building-control or engineering recommendations. A production building automation system would require real sensor data, domain-specific validation, monitoring, safety controls, and integration with appropriate building-management infrastructure.

---

# Author

Nilanga Jayawickrama

MSc Applied Computer Science  
Fairleigh Dickinson University, Vancouver, BC