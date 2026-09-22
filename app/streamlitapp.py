"""Streamlit interface for running LipNet predictions on a video."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import imageio
import streamlit as st
import tensorflow as tf

from modelutil import load_model
from utils import load_data, num_to_char

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "s1"

st.set_page_config(page_title="LipBuddy", layout="wide")


@st.cache_resource(show_spinner="Loading trained LipNet model…")
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def predict(video_path: str):
    video, _ = load_data(video_path)
    model = get_model()
    probabilities = model.predict(tf.expand_dims(video, axis=0), verbose=0)
    decoded = tf.keras.backend.ctc_decode(
        probabilities, input_length=[probabilities.shape[1]], greedy=True
    )[0][0]
    text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8").strip()
    return video.numpy(), text


def uploaded_video_path(upload) -> Path:
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload.getbuffer())
        return Path(temp_file.name)


with st.sidebar:
    st.title("LipBuddy")
    st.caption("Lip reading with the included pretrained LipNet checkpoint")
    source = st.radio("Video source", ("Upload a video", "Bundled sample"))

st.title("LipNet video lip reader")
st.caption("The complete clip is analyzed once, then its extracted text stays visible while you play it.")

video_bytes = None
video_path: Path | None = None
cleanup_path: Path | None = None

if source == "Upload a video":
    upload = st.file_uploader("Choose a video", type=("mp4", "avi", "mov", "mpg", "mpeg", "webm"))
    if upload:
        video_bytes = upload.getvalue()
        video_path = uploaded_video_path(upload)
        cleanup_path = video_path
else:
    samples = sorted(SAMPLE_DIR.glob("*.mpg")) if SAMPLE_DIR.is_dir() else []
    if not samples:
        st.error(f"No bundled samples found in {SAMPLE_DIR}")
    else:
        video_path = st.selectbox("Choose a sample", samples, format_func=lambda path: path.name)
        video_bytes = video_path.read_bytes()

if video_path:
    left, right = st.columns(2)
    with left:
        st.subheader("Video")
        st.video(video_bytes)
    try:
        with st.spinner("Reading lips from the video…"):
            frames, extracted_text = predict(str(video_path))
        with right:
            st.subheader("Extracted text")
            st.success(extracted_text or "(No text detected)")
            st.caption("Prediction is ready for the entire clip and remains visible during playback.")
            with st.expander("Model input frames"):
                gif_bytes = imageio.mimsave("<bytes>", frames.squeeze(-1).astype("uint8"), format="GIF", fps=10)
                st.image(gif_bytes)
    except Exception as error:
        with right:
            st.subheader("Extracted text")
            st.error(f"Could not analyze this video: {error}")
        st.info("For reliable results, use a clear, front-facing, well-lit video containing one speaker's mouth.")
    finally:
        if cleanup_path:
            cleanup_path.unlink(missing_ok=True)

st.divider()
st.caption(
    "This pretrained model was trained on GRID-style, front-facing mouth videos. "
    "It can run on uploaded clips, but accurate results on arbitrary camera angles require retraining "
    "or a face-landmark mouth-cropping pipeline."
)
