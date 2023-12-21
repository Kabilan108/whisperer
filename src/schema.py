# schema.py
#
# Author: Tony K. Okeke
# Created: 2023-12-21

from pydantic import BaseModel, Field, field_validator
import pandas as pd

from typing import Optional
import re


class WhisperRequest(BaseModel):
    """Request body for the Whisper model"""

    audio: str = Field(None)
    audio_base64: str = Field(None)

    transcript: str = Field("text")
    language: str = Field("en")
    timestamps: bool = Field(True)

    aggregate: bool = Field(True)
    aggregate_chunksize: int = Field(60)


class TimeStamp(BaseModel):
    start: str = Field(..., description="Start time of the segment")
    end: Optional[str] = Field(None, description="End time of the segment")

    @field_validator("start", "end")
    def validate_start_time_format(cls, v):
        if not re.match(r"^(?:[01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$", v):
            raise ValueError("Time must be in the H:MM:SS or HH:MM:SS format")
        return v

    def to_timedelta(self):
        # fmt = '%H:%M:%S'
        start_td = (
            pd.to_timedelta(self.start + ":00")
            if len(self.start.split(":")) == 2
            else pd.to_timedelta(self.start)
        )
        end_td = (
            pd.to_timedelta(self.end + ":00")
            if len(self.end.split(":")) == 2
            else pd.to_timedelta(self.end)
        )
        return start_td, end_td

    @classmethod
    def from_timedelta(cls, start_td, end_td):
        start_str = str(start_td)[-8:]
        end_str = str(end_td)[-8:]
        return cls(start=start_str, end=end_str)
