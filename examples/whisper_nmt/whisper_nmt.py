import ctranslate2
import librosa
import transformers

# Load and resample the audio file.
audio, _ = librosa.load("testaudio_16000_test01_20s.wav", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-medium")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# Load the model on CPU.
model = ctranslate2.models.WhisperNmt("models/whisper-nmt-ct2", device="cuda")

results = model.generate(features, ["en"], [["</s>"]])
print(results)
