# LipNet

A deep learning lip-reading model (based on the LipNet architecture) with a Streamlit app that predicts spoken text from silent video of a speaker's mouth.

## Project structure

```
LipNet/
├── app/
│   ├── streamlitapp.py   # Streamlit web app
│   ├── modelutil.py      # Builds the model and loads trained weights
│   └── utils.py          # Video/alignment loading and preprocessing
├── LipNet.ipynb           # Notebook used to train the model from scratch
├── models - checkpoint 50.zip   # Pretrained weights (epoch 50)
├── models - checkpoint 96.zip   # Pretrained weights (epoch 96, recommended)
└── requirements.txt
```

## Prerequisites

- Python 3.9 (the included `.pyc` cache files were built with 3.9; other 3.9–3.11 versions should also work)
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your system `PATH` (used to transcode sample videos for playback in the app)

## 1. Set up a virtual environment

From the `LipNet` directory:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Get the model weights

The app expects trained weights at `models/checkpoint` (relative to the `LipNet` folder).

Extract one of the provided checkpoint archives into a `models` folder:

```bash
# From the LipNet directory
unzip "models - checkpoint 96.zip" -d models
```

This should produce:

```
LipNet/models/checkpoint
LipNet/models/checkpoint.index
LipNet/models/checkpoint.data-00000-of-00001
```

Alternatively, you can download the same weights via the notebook (`LipNet.ipynb`), which pulls them from Google Drive with `gdown` and extracts them into `models/`.

## 4. Get the sample video data

The app lists videos from `data/s1` and their aligned transcripts from `data/alignments/s1` (relative to the `LipNet` folder). This dataset isn't included in the repo — download it by running the corresponding cell in `LipNet.ipynb`, which fetches `data.zip` via `gdown` and extracts it to `data/`, or download it manually and arrange it as:

```
LipNet/
└── data/
    ├── s1/                  # .mpg video files
    └── alignments/s1/       # matching .align transcript files
```

## 5. Run the app

From the `LipNet/app` directory:

```bash
cd app
streamlit run streamlitapp.py
```

Streamlit will open the app in your browser (default: http://localhost:8501). Select a video from the dropdown to see the original clip, the preprocessed frames the model sees, and the predicted text.

## Troubleshooting

**`AttributeError: module '...keras...layers' has no attribute 'StringLookup'`**
This means Streamlit is running with an old TensorFlow build (e.g. Anaconda's base environment) instead of the one installed in your virtual environment. `StringLookup` requires TensorFlow 2.3+. Make sure the venv is activated before running `streamlit run streamlitapp.py`, then confirm with:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

It should print 2.9 or higher and come from your `venv` path, not Anaconda's `site-packages`.

**`ValueError: File format not supported: filepath=..\models\checkpoint. Keras 3 only supports V3 .keras and .weights.h5 files...`**
The provided checkpoints are in the legacy TensorFlow checkpoint format, but TensorFlow 2.16+ ships Keras 3 by default, which dropped support for loading that format directly. The app works around this by using the `tf_keras` (Keras 2) compatibility package — make sure it's installed (`pip install -r requirements.txt` includes it) and that `TF_USE_LEGACY_KERAS=1` is set before TensorFlow is imported (already handled at the top of `streamlitapp.py`).

## (Optional) Retrain the model

Open `LipNet.ipynb` in Jupyter to download the dataset, build and train the model from scratch, and export new checkpoints.
