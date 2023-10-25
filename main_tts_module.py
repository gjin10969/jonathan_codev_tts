import asyncio
import datetime
import os
import time
import traceback
import edge_tts
import numpy as np
import soundfile as sf
import librosa
# import bark_tts as bark
from vc_infer_pipeline import VC
from models_initialize import (
    model_data,
    limitation,
    hubert_model, 
    load_hubert, 
    rmvpe_model, 
    file1_path, generate_random_filename)



from bark.api import generate_audio
from IPython.display import Audio

from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
import soundfile as sf  # Import the soundfile library

semantic_path = None # set to None if you don't want to use finetuned semantic
coarse_path = None # set to None if you don't want to use finetuned coarse
fine_path = None # set to None if you don't want to use finetuned fine
use_rvc = False # Set to False to use bark without RVC

rvc_name = 'mi-test'
rvc_path = f"Retrieval-based-Voice-Conversion-WebUI/weights/{rvc_name}.pth"
index_path = f"Retrieval-based-Voice-Conversion-WebUI/logs/{rvc_name}/added_IVF256_Flat_nprobe_1_{rvc_name}_v2.index"
device="cpu:0"
is_half=True
# download and load all models
preload_models(
    text_use_gpu=True,
    text_use_small=False,
    text_model_path=semantic_path,
    coarse_use_gpu=True,
    coarse_use_small=False,
    coarse_model_path=coarse_path,
    fine_use_gpu=True,
    fine_use_small=False,
    fine_model_path=fine_path,
    codec_use_gpu=True,
    force_reload=True,
    path="models"
)




def tts(
    model_name,
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
        SPEAKER = None
        sentences = tts_text.split("\n")

        # Remove any empty lines from the array
        sentences = [line for line in sentences if line.strip() != ""]
        print(sentences)

        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

        pieces = []
        for sentence in sentences:
            audio_array = generate_audio(sentence, history_prompt=SPEAKER, text_temp = 0.7, waveform_temp = 0.7)
            pieces += [audio_array, silence.copy()]
        Audio(np.concatenate(pieces), rate=SAMPLE_RATE)
        output_audio = np.concatenate(pieces)

        # Define the output file name
        output_file = "generated_audio.wav"

        # Save the audio as a .wav file
        sf.write(output_file, output_audio, SAMPLE_RATE)

        # Display the audio
        Audio(output_audio, rate=SAMPLE_RATE)

        print(f"Audio saved as {output_file}")
        audio, sr = librosa.load(output_file, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        if limitation and duration >= 20:
            print("Error: Audio too long")
            return (
                f"Audio should be less than 20 seconds in this huggingface space, but got {duration}s.",
                output_file,
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
            output_file,
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
        info = f"Success. npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        print(audio_opt)

        # Generate a random filename for the output
        output_vtv = generate_random_filename() + ".mp3"
        current_tts = os.path.dirname(os.path.abspath(__file__))
        folder_path_tts = os.path.join(current_tts, 'uploads')
        full_file_path = os.path.join(folder_path_tts, output_vtv)

        # Write the audio_opt to the output file
        sf.write(full_file_path, audio_opt, tgt_sr)
        os.system(f"ffmpeg -i {full_file_path} -ar 48000")

        # Create the response data dictionary
        return {
            "edge_voice": output_file,
            "audio_opt": full_file_path,
            "output_tts": file1_path + output_vtv
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

