from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    indoor_temperature: float = Field(ge=5, le=40)
    outdoor_temperature: float = Field(ge=-40, le=55)
    humidity: float = Field(ge=0, le=100)
    occupancy: int = Field(ge=0, le=500)
    hvac_power_kw: float = Field(ge=0, le=500)