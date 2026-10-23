"""Request/response models. Bounds reject physically impossible input early."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_c: float = Field(ge=-50, le=60)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="Monday = 0")
    is_holiday: int = Field(ge=0, le=1)
    wind_speed_ms: float = Field(ge=0, le=80)
    cloud_cover: float = Field(ge=0, le=1)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[Features] = Field(min_length=1)


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    predictions: list[float]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    model_uri: str
    run_id: str | None
