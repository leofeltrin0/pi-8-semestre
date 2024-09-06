# stt/audio_processing.py
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")

def split_audio(audio, sr, chunk_duration=30):
    """Split audio into chunks of `chunk_duration` seconds."""
    chunk_size = int(chunk_duration * sr)
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    
    chunks = [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    return chunks

def transcribe_audio(audio_path):
    try:
        data, samplerate = librosa.load(audio_path, sr=16000)
        audio_chunks = split_audio(data, samplerate, chunk_duration=30)
        transcriptions = []

        for chunk in audio_chunks:
            input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(transcription)
        
        return ' '.join(transcriptions)
    
    except Exception as e:
        raise RuntimeError(f"Error in transcribing audio: {str(e)}")