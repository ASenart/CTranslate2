import os
os.environ["CT2_VERBOSE"] = "1"

import ctranslate2
import librosa
import transformers
from datasets import load_dataset, Audio

# Load and resample the audio file.
dataset = load_dataset("google/fleurs", "ar_eg", trust_remote_code=True, split="test", streaming=True)
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-medium")
model = ctranslate2.models.WhisperNmt("whisper-nmt-ct2", device="cpu", intra_threads=12)

trans = load_dataset("google/fleurs", "fr_fr", trust_remote_code=True, split="test", streaming=True)

batch = []
for ind, (example, t) in enumerate(zip(dataset, trans)):
    audio = example["audio"]["array"]
    batch.append(audio)
    if len(batch)==10:
        # Compute the features of the first 30 seconds of audio.
        inputs = processor(batch, return_tensors="np", sampling_rate=16000)
        features = ctranslate2.StorageView.from_array(inputs.input_features)
        # Load the model on CPU.
        results = model.translate(features, [["｟"+"en"+"｠"]]*len(batch))
        for res in results:
            print(" ".join(res.sequences[0]))
        batch = []
        break
if batch:
    inputs = processor(batch, return_tensors="np", sampling_rate=16000)
    features = ctranslate2.StorageView.from_array(inputs.input_features)
    # Load the model on CPU.
    results = model.translate(features, [["｟"+"en"+"｠"]]*len(batch))
    for res in results:
        print(" ".join(res.sequences[0]))
    batch = []
