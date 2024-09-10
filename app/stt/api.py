from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
import uvicorn
from audio_processing import transcribe_audio, summarize

app = FastAPI()

@app.post("/transcribe/")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    try:
        # Read the content of the uploaded file
        audio_data = await file.read()
        
        # Pass the audio data to the transcription function
        transcription = transcribe_audio(BytesIO(audio_data))
        
        return {"transcription": transcription}
    
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/transcribe_and_summarize/")
async def transcribe_and_summarize_endpoint(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        
        transcription = transcribe_audio(BytesIO(audio_data))
        
        transcription = summarize(transcription)
        
        return {"transcription": transcription}
    
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
