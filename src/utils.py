# utils.py
#
# Author: Tony K. Okeke
# Created: 2023-12-21

from typing import Dict, List, Optional, Tuple
from datetime import timedelta
from pathlib import Path

import ffmpeg

from schema import TimeStamp

_SAMPLING_RATE = 16000


def str_to_path(path: str) -> Path:
    """Convert a string to a Path object."""

    if isinstance(path, str):
        return Path(path).resolve()
    return path.resolve()


def is_video(file: Path | str) -> bool:
    """Check if the file is a video file."""

    file = str_to_path(file)
    return file.suffix == ".mp4"


def seconds_to_datetime(seconds: float) -> str:
    """Convert seconds to a datetime object."""

    # Round off the milliseconds before converting to datetime

    if seconds is None:
        return None

    return (timedelta(seconds=round(seconds))).__str__()


def to_timestamp(timestamp: Tuple[float, float], chunksize: int = 60) -> TimeStamp:
    """Convert a tuple of seconds to a TimeStamp object."""

    s = seconds_to_datetime(timestamp[0])
    e = seconds_to_datetime(timestamp[1])

    if e is None:
        # add chunksize to s
        e = seconds_to_datetime(timestamp[0] + chunksize)
        return TimeStamp(start=s, end=e)

        # return TimeStamp(start=s, end=)

    return TimeStamp(start=s, end=e)


def to_csv(chunks: List[Dict]) -> str:
    """Convert a list of transcript chunks to a CSV string."""

    csv = "'start', 'end', 'text'\n"

    for seg in chunks:
        timestamp, text = seg.values()
        start, end = timestamp.values()

        csv += f"'{start}', '{end}', '{text}'\n"

    return csv


def to_tsv(chunks: List[Dict]) -> str:
    """Convert a list of transcript chunks to a TSV string."""

    tsv = "'start'\t'end'\t'text'\n"

    for seg in chunks:
        timestamp, text = seg.values()
        start, end = timestamp.values()

        tsv += f"'{start}'\t'{end}'\t'{text}'\n"

    return tsv


def aggregate_chunks(segments: List[Dict], chunksize: int = 30) -> List[Dict]:
    """Aggregate transcripts within a `chunksize` interval into a single chunk."""

    def reset_chunk():
        return "", None, timedelta()

    timestamps = [seg["timestamp"].to_timedelta() for seg in segments]
    texts = [seg["text"] for seg in segments]

    new_data = []
    current_chunk, current_start, elapsed = reset_chunk()

    for (start, end), text in zip(timestamps, texts):
        if current_start is None:
            current_start = start

        current_chunk += text
        elapsed += end - start

        if elapsed.total_seconds() >= chunksize:
            new_data.append(
                {
                    "timestamp": TimeStamp.from_timedelta(current_start, end),
                    "text": current_chunk,
                }
            )
            current_chunk, current_start, elapsed = reset_chunk()

    if current_chunk:
        new_data.append(
            {
                "timestamp": TimeStamp.from_timedelta(current_start, end),
                "text": current_chunk,
            }
        )

    return new_data


def video_to_wav(
    input_file: Path | str, output_file: Optional[Path | str] = None
) -> Path:
    """Convert a mp4 video file to a 16khz wav audio file.

    Args:
        input_file (Path): Path to the input file.
        output_file (Path): Path to the output file.

    Returns:
        Path: Path to the output file.
    """

    input_file = str_to_path(input_file)

    if output_file is None:
        output_file = input_file.with_suffix(".wav").resolve()
    output_file = str_to_path(output_file)

    assert input_file.suffix == ".mp4", "Input file must be a MP4 file"
    assert output_file.suffix == ".wav", "Output file must be a WAV file"

    try:
        _ = (
            ffmpeg.input(input_file)
            .output(
                f"{output_file}",
                format="wav",
                acodec="pcm_s16le",
                ac=1,
                ar=_SAMPLING_RATE,
            )
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        return output_file
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Failed to convert video to audio: {e.stderr.decode('utf-8')}"
        ) from e
