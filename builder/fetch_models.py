# fetch_models.py
#
# Author: Tony K. Okeke
# Created: 2023-12-21

from transformers import pipeline
import torch


def load_model():
    """Load and cache models in parallel."""

    global model_name

    if torch.cuda.is_available():
        for _attempt in range(5):
            while True:
                try:
                    torch.cuda.empty_cache()
                    pipe = pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        torch_dtype=torch.float16,
                        device="cuda:0",
                    )
                except (AttributeError, OSError):
                    continue
                except torch.cuda.OutOfMemoryError:
                    print("Model loading failed due to OOM error.")
                    break

                break
    else:
        return None, None

    return model_name, pipe


if __name__ == "__main__":
    model_name = "openai/whisper-large-v3"
    model_name, model = load_model()
