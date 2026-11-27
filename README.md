<div align="center">

# 🎙️ Speech-to-Text with OpenAI Whisper

**Audio-to-text transcription with Whisper and the Hugging Face `transformers` pipeline**

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/model-openai%2Fwhisper--base-412991)
![HF](https://img.shields.io/badge/🤗-transformers-yellow)
![Colab](https://img.shields.io/badge/runs%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

</div>

## 📌 Overview
A compact, reproducible pipeline that transcribes speech from an audio file (`.mp3`, `.wav`, `.opus` and more) to text using OpenAI's **Whisper** model. It was built as a technical assignment. The [slide deck](Converting-Speech-to-Text-Using-Whisper.pptx) explains how Whisper works and the design choices made here.

## 🧠 How Whisper works (in brief)
Whisper is an **encoder–decoder Transformer** trained on about 680k hours of multilingual audio with weak supervision.

```
Audio → 16 kHz resample → 30 s log-Mel spectrogram (80 bins)
      → Transformer encoder → Transformer decoder (autoregressive tokens)
      → Text
```

Training on such a large and varied dataset makes it robust to accents, background noise and technical vocabulary without fine-tuning.

## ⚙️ Implementation

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
result = pipe("test audio.opus")
print(result["text"])
```

**Sample output** (on the included `test audio.opus`):

> *"Hi, my name is Ashima Pandey and I study at Sharda University."*

**Model choice:** `whisper-base` (74M parameters) is a good trade-off between speed and accuracy and runs comfortably on a free Colab instance. Changing to `whisper-small` or `whisper-medium` is a one-line edit if you need higher accuracy.

## 🗂 Repository

| File | Description |
|---|---|
| `Assignment_Cyfuture.ipynb` | Transcription notebook (Colab-ready) |
| `Converting-Speech-to-Text-Using-Whisper.pptx` | Presentation on the approach |
| `test audio.opus` | Sample input audio |
| `Screenshot (110).png` | Output screenshot |

## ▶️ Run it
1. Open `Assignment_Cyfuture.ipynb` in Google Colab (or locally with Python 3.9+).
2. Install the dependencies: `pip install transformers torchaudio`. Audio decoding also needs `ffmpeg`.
3. Point `audio_path` at your own audio file and run all cells.

## 🔭 Possible extensions
- Long-form audio with `chunk_length_s=30` and timestamps
- Language detection and translation (`task="translate"`)
- Benchmark word error rate (WER) across Whisper model sizes on a public dataset such as LibriSpeech
- Wrap it in a Gradio or Streamlit app for drag-and-drop transcription
