# schema.py
#
# Author: Tony K. Okeke
# Created: 2023-12-21

from pydantic import BaseModel, Field, field_validator
import pandas as pd

from typing import List, Optional
import re


class TimeStamp(BaseModel):
    """Timestamp object."""

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

    def __str__(self):
        return f"{self.start} --> {self.end}"

    def __repr__(self):
        return f"{self.start} --> {self.end}"

    def model_dump(self):
        return {"start": self.start, "end": self.end}


class Chunk(BaseModel):
    """ "Transcript chunk."""

    timestamp: TimeStamp = Field(...)
    text: str = Field(...)

    def __str__(self):
        return f"{self.timestamp}\t{self.text}"

    def __repr__(self):
        return f"{self.timestamp}\t{self.text}"

    def model_dump(self):
        return {"timestamp": self.timestamp.model_dump(), "text": self.text}


class Transcript(BaseModel):
    """Transcript object."""

    chunks: List[Chunk] = Field(...)
    text: str = Field(...)

    def __str__(self):
        return "\n".join(str(chunk) for chunk in self.chunks)

    def __repr__(self):
        return "\n".join(str(chunk) for chunk in self.chunks)

    def model_dump(self):
        return {
            "chunks": [chunk.model_dump() for chunk in self.chunks],
            "text": self.text,
        }


class TranscriptionRequest(BaseModel):
    """Request body for the transcription endpoint."""

    audio_base64: str = Field(None)
    language: str = Field("en")
    aggregate: bool = Field(True)
    chunksize: int = Field(60)


class TranscriptionResponse(BaseModel):
    """Response body for the transcription endpoint."""

    transcript: Optional[Transcript] = Field(None)
    errors: Optional[List[str]] = Field(None)
