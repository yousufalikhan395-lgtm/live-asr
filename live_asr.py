import time, queue, threading, collections
import numpy as np
import sounddevice as sd
import onnx_asr
from pynput import keyboard

SR            = 16000
VAD_CHUNK     = 512      # silero requires exactly 512 @16k
END_SILENCE   = 0.35     # was 0.7 — biggest "feels slow" culprit
VAD_THRESHOLD = 0.5
MIN_SPEECH    = 0.25     # ignore coughs / clicks
PREROLL_SEC   = 0.3      # audio kept from BEFORE speech started
MAX_UTTERANCE = 20.0

print("📥 Loading int8 ONNX Parakeet...")
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")

print("🔥 Warming up...")
model.recognize(np.zeros(SR, dtype=np.float32), sample_rate=SR)

# VAD (tiny, ~1 ms per chunk)
import torch
torch.set_num_threads(1)
vad, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad.eval()

kb = keyboard.Controller()
q  = queue.Queue()

preroll   = collections.deque(maxlen=int(PREROLL_SEC * SR / VAD_CHUNK))
speech    = []
speaking  = False
silence_t = 0.0
speech_t  = 0.0

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        audio, wall = item
        t0 = time.perf_counter()
        text = model.recognize(audio, sample_rate=SR).strip()
        dt = time.perf_counter() - t0
        if text:
            print(f"\n💬 {text}   [{dt:.2f}s for {len(audio)/SR:.1f}s audio, "
                  f"{len(audio)/SR/dt:.0f}× realtime, {wall:.2f}s after you stopped]")
            kb.type(text)
        print("✅ listening...", end="\r", flush=True)
        q.task_done()

threading.Thread(target=worker, daemon=True).start()

def callback(indata, frames, t, status):
    global speaking, silence_t, speech_t, speech
    chunk = indata[:, 0].copy()

    with torch.no_grad():
        p = vad(torch.from_numpy(chunk).unsqueeze(0), SR).item()

    if p > VAD_THRESHOLD:
        if not speaking:
            speaking = True
            speech = list(preroll)      # grab the 300 ms before you started
            speech_t = 0.0
            print("🎤 hearing you...", end="\r", flush=True)
        speech.append(chunk)
        speech_t += len(chunk) / SR
        silence_t = 0.0
    else:
        preroll.append(chunk)
        if speaking:
            speech.append(chunk)
            silence_t += len(chunk) / SR
            if silence_t >= END_SILENCE or speech_t >= MAX_UTTERANCE:
                speaking = False
                if speech_t >= MIN_SPEECH:
                    q.put((np.concatenate(speech), silence_t))
                speech = []
                print("⌨️ transcribing...", end="\r", flush=True)

stream = sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        callback=callback, blocksize=VAD_CHUNK)
stream.start()
print("\n✅ ALWAYS ON — just talk. Ctrl+C to quit.\n")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n🛑 stopping")
finally:
    q.put(None)
    stream.stop(); stream.close()