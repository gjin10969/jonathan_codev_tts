import asyncio
import os
import logging
import nest_asyncio
import edge_tts
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Annotated
import cachetools
import itertools

from config import Config  # Import the specific variables or classes you need from config
from vc_infer_pipeline import VC
from main_tts_module import tts
from main_vtv_module import vtv
from sqlalchemy.orm import Session
from database_config import SessionLocal, engine, Base
from database_model import Post

# Create a cache with a maximum size (you can adjust the max_size as needed)
cache_vtv = cachetools.LRUCache(maxsize=1000)  # LRUCache with a maximum size of 1000 items
request_counter = itertools.count()

Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Initialize Jinja2Templates for HTML rendering
templates = Jinja2Templates(directory="templates")
nest_asyncio.apply()

# Set logging levels for various modules
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Determine if running on a system with limitations
limitation = os.getenv("SYSTEM") == "spaces"

# Load configuration
config = Config()

# Get a list of available TTS voices
tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]

# Root directory for model weights
model_root = "weights"

# List available models
models = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]

# Check if any models are available
if len(models) == 0:
    raise ValueError("No model found in `weights` folder")

# Sort the list of models
models.sort()

# Define a cache key function (you can modify this as needed)
def get_cache_key_vtv(model_name, input_audio, f0_up_key, f0_method, index_rate, protect, filter_radius, resample_sr, rms_mix_rate):
    request_count = next(request_counter)  # Increment the request counter
    return f"{model_name}-{input_audio.filename}-{f0_up_key}-{f0_method}-{index_rate}-{protect}-{filter_radius}-{resample_sr}-{rms_mix_rate}-{request_count}"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

# Endpoint to download a file
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = filename
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

# Pydantic model for text-to-speech input
class TextToSpeechInput(BaseModel):
    model_name: str = 'eli'
    tts_text: str = 'Greetings, I am Eli, and I warmly introduce myself with the sound of my voice.'
    tts_voice: str = 'ja-JP-NanamiNeural-Female'
    f0_up_key: int = 0
    f0_method: str = 'rmvpe'
    index_rate: float = 0.5
    protect: float = 0.33

# Endpoint for text-to-speech
@app.post("/tts/")
async def tts_endpoint(data: TextToSpeechInput):
    result = await asyncio.to_thread(
        tts,
        model_name=data.model_name,
        tts_text=data.tts_text,
        tts_voice=data.tts_voice,
        f0_up_key=data.f0_up_key,
        f0_method=data.f0_method,
        index_rate=data.index_rate,
        protect=data.protect,
    )
    return result

# Endpoint for voice-to-voice conversion
@app.post("/vtv/")
async def voice_conversion(
    input_audio: UploadFile = File(...),
    model_name: str = Form(),
    f0_up_key: int = 0,
    f0_method: str = 'rmvpe',
    index_rate: float = 0.5,
    protect: float = 0.33,
    filter_radius: int = 3,
    resample_sr: int = 48000,
    rms_mix_rate: float = 0.25,
    db: Session = Depends(get_db)  # Inject the database session
):
    cache_key_vtv = get_cache_key_vtv(model_name, input_audio, f0_up_key, f0_method, index_rate, protect, filter_radius, resample_sr, rms_mix_rate)

    # Check if the result is already in the cache
    cached_result_vtv = cache_vtv.get(cache_key_vtv)
    if cached_result_vtv:
        return cached_result_vtv

    try:
        # Call your vtv function here
        result = await asyncio.to_thread(
            vtv,
            model_name,
            input_audio,
            f0_up_key,
            f0_method,
            index_rate,
            protect,
            filter_radius,
            resample_sr,
            rms_mix_rate
        )

        new_post = Post(
            input_audio=input_audio.filename,
            model_name=model_name,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            result_=result
        )
        db.add(new_post)
        db.commit()

        response_data = {
            "status": "success",
            "model_name": model_name,
            "f0_up_key": f0_up_key,
            "f0_method": f0_method,
            "index_rate": index_rate,
            "protect": protect,
            "filter_radius": filter_radius,
            "resample_sr": resample_sr,
            "rms_mix_rate": rms_mix_rate,
            "result_": result
        }

        # Store the result in the cache
        cache_vtv[cache_key_vtv] = JSONResponse(content=response_data)

        return JSONResponse(content=response_data)
    except Exception as e:
        # Handle errors and return an error response
        error_message = str(e)
        response_data = {"status": "error", "error_message": error_message}
        return JSONResponse(content=response_data, status_code=500)

# Endpoint to retrieve user information by ID
@app.get("/vtv/{id}", status_code=status.HTTP_200_OK)
async def read_user(id: int, db: db_dependency):
    user = db.query(Post).filter(Post.id == id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Endpoint to delete a post by ID
@app.delete("/posts/{id}", status_code=status.HTTP_200_OK)
async def delete_post(id: int, db: db_dependency):
    user = db.query(Post).filter(Post.id == id).first()
    if user is None:
        raise HTTPException(status_code=404, detail='Post was not found')
    db.delete(user)
    db.commit()

# Endpoint for the TTS HTML page
@app.get("/speech_text")
async def index(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request, "models": models, "tts_voices": tts_voices})

# Endpoint for the VTV HTML page
@app.get("/voice_voice")
async def index2(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request, "models": models})

# Root endpoint
@app.get("/")
async def index2(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": models})
# Run the FastAPI app
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(
#                app,
#                host="0.0.0.0",
#                port=8432,
#                ssl_keyfile="./localhost+4-key.pem",
#                ssl_certfile="./localhost+4.pem"
#                )