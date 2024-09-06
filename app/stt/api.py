# stt/api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from io import BytesIO
import uvicorn

from audio_processing import transcribe_audio

app = FastAPI()

class Data(BaseModel):
    audio_path: str

@app.post("/transcribe/")
async def transcribe_audio_endpoint(input: Data):
    try:
        transcription = transcribe_audio(input.audio_path)
        return {"transcription": transcription}
    
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
