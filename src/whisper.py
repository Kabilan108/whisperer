# predict.py
#
# Author: Tony K. Okeke
# Created: 2023-12-21

from runpod.serverless.utils import rp_cuda
from transformers import pipeline
import torch

from typing import Any
import logging

from schema import TranscriptionResponse
import utils

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "openai/whisper-large-v3"
DEVICE = "cuda:0" if rp_cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if rp_cuda.is_available() else torch.int8


class Whisper:
    """Predictor class for the Whisper model

    This predictor wraps the automatic-speech-recognition pipeline from
    transformers. It includes methods for loading the model and making it work
    on the RunPod platform.
    """

    def __init__(self):
        self.models = {}

    def load_model(self):
        """Load the mdoel into a pipeline."""

        _pipeline = pipeline(
            "automatic-speech-recognition",
            model=MODEL,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )

        return _pipeline

    def setup(self):
        """Load the model into memory."""
        self.models[MODEL] = self.load_model()

    def transcribe(
        self,
        audiofile,
        language: str = "en",
        aggregate: bool = True,
        chunksize: int = 30,
    ) -> Any:
        """Transcribe an audio file.

        Args:
            audiofile (str): The audio file to be transcribed. Must be a path to a
                file on disk.
            language (str): The language of the audio file. Must be a valid
                language code for one of the 99 languages supported Whisper.
            aggregate (bool): Whether to aggregate the transcript into chunks of
                `chunksize` seconds.
            chunksize (int): The chunksize in seconds.
        """

        whisper = self.models[MODEL]
        error = None

        if not whisper:
            raise ValueError("Model not found.")

        try:
            segments = whisper(
                audiofile,
                chunk_length_s=30,
                batch_size=12,
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": language,
                },
            )
            torch.cuda.empty_cache()
        except Exception as e:
            error = {
                "message": "Unable to transcribe audio file.",
                "details": str(e),
            }

        if error:
            result = TranscriptionResponse(error=error)
        else:
            # convert timestamp strings to TimeStamp objects
            for i, seg in enumerate(segments["chunks"]):
                timestamp, text = seg.values()

                segments["chunks"][i] = {
                    "timestamp": utils.to_timestamp(timestamp),
                    "text": text,
                }

            if aggregate:
                segments["chunks"] = utils.aggregate_chunks(
                    segments["chunks"], chunksize=chunksize
                )

            result = TranscriptionResponse(transcript=utils.to_transcript(segments))

        return result
