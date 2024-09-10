import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from io import BytesIO
from llm.model_pipeline import TextGenerationPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")
llm_model = TextGenerationPipeline("meta-llama/Meta-Llama-3.1-8B-Instruct")

def split_audio(audio, sr, chunk_duration=30):
    """Split audio into chunks of `chunk_duration` seconds."""
    chunk_size = int(chunk_duration * sr)
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    
    chunks = [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    return chunks

def transcribe_audio(audio_file: BytesIO):
    try:
        # Load the audio from the BytesIO object
        data, samplerate = librosa.load(audio_file, sr=16000)
        
        # Split audio into chunks
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
    
def summarize(msg: str):
    return llm_model.generate_text(msg)
