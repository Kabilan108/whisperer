# fetch_models.py
#
# Author: Tony K. Okeke
# Created: 2023-12-21


from huggingface_hub import snapshot_download

MODEL = "openai/whisper-large-v3"


if __name__ == "__main__":
    snapshot_download(MODEL)
