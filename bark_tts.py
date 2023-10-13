# from bark.api import generate_audio
# from IPython.display import Audio

# from transformers import BertTokenizer
# from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
# import soundfile as sf  # Import the soundfile library

# semantic_path = None # set to None if you don't want to use finetuned semantic
# coarse_path = None # set to None if you don't want to use finetuned coarse
# fine_path = None # set to None if you don't want to use finetuned fine
# use_rvc = False # Set to False to use bark without RVC

# rvc_name = 'mi-test'
# rvc_path = f"Retrieval-based-Voice-Conversion-WebUI/weights/{rvc_name}.pth"
# index_path = f"Retrieval-based-Voice-Conversion-WebUI/logs/{rvc_name}/added_IVF256_Flat_nprobe_1_{rvc_name}_v2.index"
# device="cpu:0"
# is_half=True
# # download and load all models
# preload_models(
#     text_use_gpu=True,
#     text_use_small=True,
#     text_model_path=semantic_path,
#     coarse_use_gpu=True,
#     coarse_use_small=True,
#     coarse_model_path=coarse_path,
#     fine_use_gpu=True,
#     fine_use_small=True,
#     fine_model_path=fine_path,
#     codec_use_gpu=True,
#     force_reload=False,
#     path="models"
# )


# import numpy as np
# # torch.manual_seed(1)

# # simple generation
# tts_text = """
# 雨の日も風の日も、新聞配達さんは夜明けが早い。
# """
# # SPEAKER = "/content/bark-with-voice-clone/bark/assets/prompts/0.npz" # use your custom voice name here if you have on
# # SPEAKER = "output"
# SPEAKER = None
# sentences = tts_text.split("\n")

# # Remove any empty lines from the array
# sentences = [line for line in sentences if line.strip() != ""]
# print(sentences)

# silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

# pieces = []
# for sentence in sentences:
#     audio_array = generate_audio(sentence, history_prompt=SPEAKER, text_temp = 0.7, waveform_temp = 0.7)
#     pieces += [audio_array, silence.copy()]
# Audio(np.concatenate(pieces), rate=SAMPLE_RATE)
# output_audio = np.concatenate(pieces)

# # Define the output file name
# output_file = "generated_audio.wav"

# # Save the audio as a .wav file
# sf.write(output_file, output_audio, SAMPLE_RATE)

# # Display the audio
# Audio(output_audio, rate=SAMPLE_RATE)

# print(f"Audio saved as {output_file}")