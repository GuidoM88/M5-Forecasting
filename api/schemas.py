"""Validated request schema."""
from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints, field_validator


class PredictionRequest(BaseModel):
    item_ids: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = Field(
        min_length=1, max_length=1000)

    @field_validator('item_ids')
    @classmethod
    def unique_items(cls, items):
        if len(set(items)) != len(items):
            raise ValueError('item_ids must be unique')
        return items
