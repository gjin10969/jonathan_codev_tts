import asyncio
import datetime
import random
import string
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydub import AudioSegment
from fastapi.responses import JSONResponse

import io
import time
import traceback
import asyncio
import datetime
import logging
import json
import edge_tts
import soundfile as sf
from fastapi.responses import FileResponse
import librosa
import torch
from fairseq import checkpoint_utils

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC
import edge_tts
import librosa
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fairseq import checkpoint_utils
from fastapi.staticfiles import StaticFiles

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC

app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)



limitation = os.getenv("SYSTEM") == "spaces"

config = Config()
file1_path = "http://0.0.0.0:8000/uploads/"

edge_output_filename = "edge_output.mp3"
tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]

model_root = "weights"
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if len(models) == 0:
    raise ValueError("No model found in `weights` folder")
models.sort()



# @app.get("/", response_class=HTMLResponse)

# Function to generate a random filename
def generate_random_filename(length=10):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def model_data(model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")
    pth_path = pth_files[0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


print("Loading hubert model...")
hubert_model = load_hubert()
print("Hubert model loaded.")

print("Loading rmvpe model...")
rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
print("rmvpe model loaded.")


def tts(
    model_name,
    speed,
    tts_text,
    tts_voice,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    filter_radius=3,
    resample_sr=48000,
    rms_mix_rate=0.25,
):
    print("------------------")
    print(datetime.datetime.now())
    print("tts_text:")
    print(tts_text)
    print(f"tts_voice: {tts_voice}")
    print(f"Model name: {model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:
        if limitation and len(tts_text) > 280:
            print("Error: Text too long")
            return (
                f"Text characters should be at most 280 in this huggingface space, but got {len(tts_text)} characters.",
                None,
                None,
            )
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        t0 = time.time()
        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"
        asyncio.run(
            edge_tts.Communicate(
                tts_text, "-".join(tts_voice.split("-")[:-1]), rate=speed_str
            ).save(edge_output_filename)
        )
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        if limitation and duration >= 20:
            print("Error: Audio too long")
            return (
                f"Audio should be less than 20 seconds in this huggingface space, but got {duration}s.",
                edge_output_filename,
                None,
            )

        f0_up_key = int(f0_up_key)

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        print(audio_opt)
####randomize
        current_tts = os.path.dirname(os.path.abspath(__file__))
        folder_path_tts= os.path.join(current_tts, 'uploads')        

        output_vtv = generate_random_filename() + ".mp3"
        

        full_file_path = os.path.join(folder_path_tts, output_vtv)

        sf.write(full_file_path, audio_opt, tgt_sr)
        os.system(f"ffmpeg -i {full_file_path} -ar 48000")

        # Create the response data dictionary
        return {
         
            "edge_voice": edge_output_filename,
            
            "audio_opt": full_file_path,
            "output_tts": file1_path + output_vtv  # Include the output_tts filename

        
        }
        
    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None
    
def vtv(
    model_name,
    input_audio_path,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    filter_radius,
    resample_sr,
    rms_mix_rate,
):
    global full_file_path, audio_opt, output_vtv

    print("------------------")
    print(datetime.datetime.now())
    print("tts_text:")

    print(f"Model name: {model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:

        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        t0 = time.time()


        audio_data = input_audio_path.file.read()

        # Convert bytes to AudioSegment
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

        # Export as MP3
        inputed_audio = "output_voice.mp3"
        audio_segment.export(inputed_audio, format="mp3")
        print("tanignaaaaaaaaaaaa")
        print(inputed_audio)

        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(inputed_audio, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        if limitation and duration >= 20:
            print("Error: Audio too long")
            return (
                f"Audio should be less than 20 seconds in this huggingface space, but got {duration}s.",
                inputed_audio,
                None,
            )
        print(inputed_audio)
        print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

        f0_up_key = int(f0_up_key)

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            inputed_audio,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        print(audio_opt)
        print(inputed_audio)
##############################################33
        # length = 4
        # letters = string.ascii_lowercase
        # result_str = ''.join(random.choice(letters) for i in range(length))
        

        # output_tts = result_str + ".mp3"
        # print(output_tts)

        #audio file format
        current_directory = os.path.dirname(os.path.abspath(__file__))

        folder_path = os.path.join(current_directory, 'uploads')
##mp3 converter
        output_vtv = generate_random_filename() + ".mp3"
        # output_tts = "output.mp3"

        print(output_vtv)
        full_file_path = os.path.join(folder_path, output_vtv)
        

        #audio file format
        # output_wav_path = "output.mp3"
        sf.write(full_file_path, audio_opt, tgt_sr)
        os.system(f"ffmpeg -i {full_file_path} -ar 48000")

        # Create the response data dictionary
        # final_output = file1_path + output_vtv
        # return final_output
        return { 
            "inputed_audio": inputed_audio,      
            "audio_opt": full_file_path,
            "output_vtv": file1_path + output_vtv  # Include the output_tts filename

        }

    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None

#################################
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = filename
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

############################# TEXT TO SPEECH ################

class TextToSpeechInput(BaseModel):


    model_name: str = 'eli'
    speed: int = 0
    tts_text: str = 'Greetings, I am Eli, and I warmly introduce myself with the sound of my voice.'
    tts_voice: str = 'ja-JP-NanamiNeural-Female'
    f0_up_key: int = 0
    f0_method: str = 'rmvpe'
    index_rate: float = 0.5
    protect: float = 0.33

@app.post("/tts/")
async def tts_endpoint(data: TextToSpeechInput):
    print(TextToSpeechInput)
    result = await asyncio.to_thread(
        tts,
        data.model_name,
        data.speed,
        data.tts_text,
        data.tts_voice,
        data.f0_up_key,
        data.f0_method,
        data.index_rate,
        data.protect,
    )

    return result




#ROOT 

################ VOICE TO VOICE ################
    
@app.post("/vtv/")
async def voice_conversion(
    input_audio_path: UploadFile = File(...),
    model_name: str = Form(),
    f0_up_key: int = 0,
    f0_method: str = 'rmvpe',
    index_rate: float = 0.5,
    protect: float = 0.33,
    filter_radius: int = 3,
    resample_sr: int = 48000,
    rms_mix_rate: float = 0.25
):
    try:
        result = await asyncio.to_thread( 
            vtv, 
            model_name,
            input_audio_path,
            f0_up_key,
            f0_method,
            index_rate,
            protect, 
            filter_radius, 
            resample_sr,
            rms_mix_rate,

        )
        #global variable
        # full_file_path,
        # audio_opt, 
        # output_vtv,
        
        # print(input_audio_path)



        # Create a JSON response with your result
        response_data = {"status": "success",
            # "audio_opt": audio_opt,
            # "full_file_path":full_file_path,
            "model_name": model_name,
            "f0_up_key": f0_up_key,
            "f0_method": f0_method,
            "index_rate": index_rate,
            "protect": protect,
            "filter_radius": filter_radius,
            "resample_sr": resample_sr,
            "rms_mix_rate": rms_mix_rate,
            "result_": result,
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        # Handle errors and return an error response
        error_message = str(e)
        response_data = {"status": "error", "error_message": error_message}
        return JSONResponse(content=response_data, status_code=500)
    
    

    # except Exception as e:
    #     error_message = str(e)

    #     return templates.TemplateResponse("index2.html", {"error_message": error_message})



    





##################HTML REQUEST################33


@app.get("/speech_text")
async def index(request: Request):
    print(models)
    return templates.TemplateResponse("index1.html", {"request": request, "models": models, "tts_voices": tts_voices})

@app.get("/voice_voice")
async def index2(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request, "models": models})


@app.get("/")
async def index2(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": models})





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)