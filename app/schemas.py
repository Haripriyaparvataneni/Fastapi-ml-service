"""Pydantic schemas for the demand-forecasting API."""

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Input features for a single demand forecast."""

    price: float = Field(..., gt=0, description="Unit price of the product")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday .. 6=Sunday")
    month: int = Field(..., ge=1, le=12, description="Calendar month")
    promotion: bool = Field(False, description="Whether a promotion is active")
    stock_on_hand: int = Field(..., ge=0, description="Units currently in stock")


class ForecastResponse(BaseModel):
    """Forecast result returned to the caller."""

    predicted_demand: float = Field(..., description="Predicted units sold")
    model_version: str = Field(..., description="Version of the served model")
