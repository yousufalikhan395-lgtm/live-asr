# 🎙️ live-asr: Always-On AI Voice Typing

A blazing-fast, privacy-focused voice-to-text assistant that lives in your terminal. It listens continuously, detects when you speak, and types your words into **any** application instantly.

Powered by **NVIDIA’s Parakeet TDT** (for speed) and **Silero VAD** (for smart silence detection), it supports English and 24+ European languages with punctuation and capitalization.

> **✨ No hotkeys required.** Just click into a text field and start talking. It handles the rest.

---

## 🚀 Features

*   **🔥 Blazing Fast:** Uses quantized (int8) ONNX models for near-instant transcription on CPU.
*   **👂 Smart Listening:** Uses Voice Activity Detection (VAD) to ignore background noise and only type when you’re actually speaking.
*   **⌨️ Universal Typing:** Works in **every** app — Discord, VS Code, Chrome, Word, Notepad, and more.
*   **🌍 Multilingual:** Supports English, Spanish, French, German, Russian, and 20+ other languages with automatic language detection.
*   **🔒 Private & Local:** All processing happens on your machine. No audio is ever sent to the cloud.
*   **🧠 Pre-roll Buffer:** Keeps the 300 ms of audio *before* you started talking, so the first syllables are never clipped.
*   **✅ Punctuation & Capitalization:** Automatically adds commas, periods, and capital letters based on your tone.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10 or higher
*   A microphone
*   Windows, Linux, or macOS

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/live-asr.git
cd live-asr
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note: The first time you run the app, it will automatically download the ~600 MB Parakeet model, and the Silero VAD model via `torch.hub`.)*

---

## ⚡ Usage

1.  **Start the Assistant:**
    ```bash
    python live_asr.py
    ```
2.  **Wait for the ready message.** You’ll see `✅ ALWAYS ON — just talk.` in your terminal.
3.  **Click into any text field** (e.g., a browser search bar, a document, or a chat window).
4.  **Speak naturally.**
    *   The app shows `🎤 hearing you...` when it detects speech.
    *   When you pause for ~0.35 s, it types the result into the focused window.
5.  **Stop:** Press `Ctrl + C` in the terminal to close the application.

---

## ⚙️ Configuration

You can tweak the sensitivity in `live_asr.py` to fit your environment:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VAD_THRESHOLD` | `0.5` | VAD probability threshold. Lower = more sensitive (picks up quiet speech). Higher = ignores background noise. |
| `END_SILENCE` | `0.35` | Seconds of silence after you stop speaking before the utterance is transcribed and typed. |
| `MIN_SPEECH` | `0.25` | Minimum speech duration (seconds) required to bother transcribing. Ignores coughs and clicks. |
| `PREROLL_SEC` | `0.3` | Seconds of audio kept *before* speech starts, so the beginning of an utterance isn't clipped. |
| `MAX_UTTERANCE` | `20.0` | Hard cap (seconds) on a single utterance before it is force-transcribed. |
| `VAD_CHUNK` | `512` | Samples per audio chunk. Silero VAD requires exactly 512 at 16 kHz. |

---

## 🧠 How It Works

1.  **Audio Capture:** Captures raw audio from your microphone at 16 kHz using `sounddevice`.
2.  **Voice Activity Detection (VAD):** Uses the lightweight **Silero VAD** model to distinguish human speech from background noise in real time (~1 ms per 32 ms chunk).
3.  **Transcription:** Once speech ends, the buffered audio (including the pre-roll) is sent to the **NVIDIA Parakeet TDT 0.6B** model, running via ONNX Runtime in int8, for high-speed transcription on a background thread.
4.  **Input Simulation:** Uses `pynput` to type the recognized text into your currently focused window.

---

## 📝 Requirements

```text
numpy
sounddevice
torch
torchaudio
onnx-asr
onnxruntime
pynput
```

---

## ⚠️ Known Limitations

*   **Language Support:** Optimized for **English and European languages**. It does not currently support Hindi, Chinese, or Japanese with the Parakeet v3 model.
*   **Admin Apps:** Applications running elevated (as Administrator) may block simulated keystrokes for security reasons. Run the terminal as Administrator if typing doesn't work in such apps.
*   **Focus Required:** Text is typed into whatever window has keyboard focus when transcription finishes — make sure the right window is focused.
*   **Latency metric:** The `wall` value printed after each utterance is the accumulated trailing silence, not the end-to-end latency.

---

## 🤝 Contributing

Pull requests are welcome! If you have ideas for better VAD tuning, support for more languages, or a GUI wrapper, please open an issue or submit a PR.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

### 💡 Pro-Tip
If transcription feels slow, try installing `onnxruntime-gpu` instead of `onnxruntime` (NVIDIA GPU required), and make sure `torch.set_num_threads(1)` in `live_asr.py` matches your setup.
