"""Video preprocessing and token conversion used by the LipNet app."""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

FRAME_COUNT = 75
MOUTH_HEIGHT = 46
MOUTH_WIDTH = 140
VOCAB = "abcdefghijklmnopqrstuvwxyz'?!123456789 "

_char_to_num_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(list(VOCAB)), tf.constant(range(1, len(VOCAB) + 1), dtype=tf.int64)
    ), default_value=0,
)
_num_to_char_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(range(1, len(VOCAB) + 1), dtype=tf.int64), tf.constant(list(VOCAB))
    ), default_value="",
)


def char_to_num(chars):
    return _char_to_num_table.lookup(tf.cast(chars, tf.string))


def num_to_char(numbers):
    return _num_to_char_table.lookup(tf.cast(numbers, tf.int64))


def _mouth_crop(frame: np.ndarray) -> np.ndarray:
    """Return the GRID-dataset mouth region, scaled for the input resolution."""
    height, width = frame.shape[:2]
    y1, y2 = round(height * 190 / 288), round(height * 236 / 288)
    x1, x2 = round(width * 80 / 360), round(width * 220 / 360)
    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if crop.size == 0:
        raise ValueError("Could not find the expected mouth area in this video frame.")
    return cv2.resize(crop, (MOUTH_WIDTH, MOUTH_HEIGHT), interpolation=cv2.INTER_AREA)


def _sample_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        raise ValueError("No readable video frames were found.")
    indices = np.linspace(0, len(frames) - 1, FRAME_COUNT).round().astype(int)
    return [frames[index] for index in indices]


def load_video(path: str | Path) -> tf.Tensor:
    """Decode, sample, crop, and normalize a video for the trained LipNet model."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video. Try MP4, AVI, MOV, or MPG.")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(_mouth_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    finally:
        capture.release()

    sampled = np.asarray(_sample_frames(frames), dtype=np.float32)[..., np.newaxis]
    mean, std = float(sampled.mean()), float(sampled.std())
    if std < 1e-6:
        raise ValueError("The video has no usable visual variation for lip reading.")
    return tf.convert_to_tensor((sampled - mean) / std, dtype=tf.float32)


def load_alignments(path: str | Path) -> tf.Tensor:
    tokens: list[str] = []
    with Path(path).open("r", encoding="utf-8") as alignment_file:
        for line in alignment_file:
            columns = line.split()
            if len(columns) >= 3 and columns[2] != "sil":
                tokens.extend((" ", columns[2]))
    return char_to_num(tf.strings.unicode_split(tokens, input_encoding="UTF-8").flat_values)[1:]


def load_data(video_path: str | Path, alignment_path: str | Path | None = None):
    """Load a video and, when present, its optional dataset alignment."""
    video = load_video(video_path)
    alignments = load_alignments(alignment_path) if alignment_path and Path(alignment_path).is_file() else None
    return video, alignments
