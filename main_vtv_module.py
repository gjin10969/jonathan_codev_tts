import asyncio
import datetime
import random
import string
import os
import io
import time
import traceback
import logging
import nest_asyncio
import json
import edge_tts
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydub import AudioSegment
from fairseq import checkpoint_utils



from models_initialize import model_data, limitation, hubert_model, load_hubert, rmvpe_model, file1_path, generate_random_filename



# Function for voice-to-voice (VTV) conversion
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

        # Generate a random filename for the output
        output_vtv = generate_random_filename() + ".mp3"
        current_directory = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(current_directory, 'uploads')

        full_file_path = os.path.join(folder_path, output_vtv)

        # Write the audio_opt to the output file
        sf.write(full_file_path, audio_opt, tgt_sr)
        os.system(f"ffmpeg -i {full_file_path} -ar 48000")

        # Create the response data dictionary
        final_out = file1_path + output_vtv
        return final_out
        # return {
        #     "inputed_audio": inputed_audio,
        #     "audio_opt": full_file_path,
        #     "output_vtv": file1_path + output_vtv,
        # }
        

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

