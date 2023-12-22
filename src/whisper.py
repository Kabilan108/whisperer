# predict.py
#
# Author: Tony K. Okeke
# Created: 2023-12-21

from runpod.serverless.utils import rp_cuda
from transformers import pipeline
import torch

from typing import Any
import logging

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

        # TODO: use bettertransformer

        return _pipeline

    def setup(self):
        """Load the model into memory."""
        self.models[MODEL] = self.load_model()

    def transcribe(
        self,
        audiofile,
        transcript: str = "plain_text",
        language: str = "en",
        timestamps: bool = True,
        aggregate: bool = True,
        chunksize: int = 30,
    ) -> Any:
        """Transcribe an audio file.

        Args:
            audiofile (str): The audio file to be transcribed. Must be a path to a
                file on disk.
            transcript (str): The type of transcript to return.
            language (str): The language of the audio file. Must be a valid
                language code for one of the 99 languages supported Whisper.
            timestamps (bool): Whether to return timestamps.
            aggregate (bool): Whether to aggregate the transcript into chunks of
                `chunksize` seconds.
            chunksize (int): The chunksize in seconds.
        """
        whisper = self.models[MODEL]

        if not whisper:
            raise ValueError("Model not found.")

        segments = whisper(
            audiofile,
            chunk_length_s=30,
            batch_size=12,
            return_timestamps=timestamps,
            generate_kwargs={
                "task": "transcribe",
                "language": language,
            },
        )

        torch.cuda.empty_cache()

        if transcript == "plain_text":
            return segments["text"]

        if timestamps:
            # convert timestamp strings to TimeStamp objects
            clean_segments = []
            for i, seg in enumerate(segments["chunks"]):
                timestamp, text = seg.values()

                clean_segments.append(
                    {"timestamp": utils.to_timestamp(timestamp), "text": text}
                )
                segments["chunks"][i] = {
                    "timestamp": utils.to_timestamp(timestamp).model_dump(),
                    "text": text,
                }

            if aggregate:
                clean_segments = utils.aggregate_chunks(
                    clean_segments, chunksize=chunksize
                )
                segments["chunks"] = [
                    {"timestamp": seg["timestamp"].model_dump(), "text": seg["text"]}
                    for seg in clean_segments
                ]

            if transcript == "text":
                return "\n".join(
                    f"{seg['timestamp']}\t{seg['text']}" for seg in clean_segments
                )
            elif transcript == "json":
                return segments["chunks"]
            elif transcript == "csv":
                return utils.to_csv(segments["chunks"])
            elif transcript == "tsv":
                return utils.to_tsv(segments["chunks"])
            else:
                raise ValueError("Unknown transcript type.")

        else:
            if transcript == "text":
                return segments["text"]
            else:
                raise ValueError(
                    "This transcript type is only available with timestamps."
                )
