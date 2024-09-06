from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
from io import BytesIO
import uvicorn
from pydantic import BaseModel
import librosa
import numpy as np

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")

class Data(BaseModel):
    audio_path: str

def split_audio(audio, sr, chunk_duration=30):
    """Split audio into chunks of `chunk_duration` seconds."""
    chunk_size = int(chunk_duration * sr)
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    
    chunks = [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    return chunks

@app.post("/transcribe/")
async def transcribe_audio(input: Data):
    try:
        data, samplerate = librosa.load(input.audio_path, sr=16000)

        audio_chunks = split_audio(data, samplerate, chunk_duration=30)

        transcriptions = []

        for chunk in audio_chunks:
            input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(transcription)
        
        full_transcription = ' '.join(transcriptions)
        
        return {"transcription": full_transcription}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
