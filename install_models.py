!apt -y install -qq aria2


!curl -L -O https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
!curl -L -O https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt

#speaker_prompts
!gdown --folder https://drive.google.com/drive/folders/1K7KsjeN-8FTfqlnBhQps7v-Aetg_degM?usp=drive_link...somefileid.. -O //root/jonathan_codev_tts/bark/assets/prompts

#semantic
!gdown --folder https://drive.google.com/drive/folders/1-7WaAZeqayqhyxbsnYc5BHO8LCcaMEJe?usp=drive_link...somefileid.. -O /root/jonathan_codev_tts/bark-model

#coarse
!gdown --folder https://drive.google.com/drive/folders/1-0qb7MivBo2HSc2OZPZLMO2nqhSC0sOM?usp=drive_link...somefileid.. -O /root/jonathan_codev_tts/bark-model

#fine
!gdown --folder https://drive.google.com/drive/folders/1tLwY4wTgASpA6CVxGKduZ8Qxso0Rwtnb?usp=sharing...somefileid.. -O /root/jonathan_codev_tts/bark-model

!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gjin10969/rvc-model-sample/resolve/main/eli/eli_sterio.pth -d /root/jonathan_codev_tts/weights/eli -o eli_sterio.pth
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gjin10969/rvc-model-sample/resolve/main/Taka/Taka_last.pth -d /root/jonathan_codev_tts/weights/Taka -o Taka_last.pth
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gjin10969/rvc-model-sample/resolve/main/hanayo/hanayo.pth -d /root/jonathan_codev_tts/weights/hanayo -o hanayo.pth
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gjin10969/rvc-model-sample/resolve/main/Aoyagisan/AoyagiVoice2YS.Speech.pth -d /root/jonathan_codev_tts/weights/Aoyagisan -o Aoyagisan.pth
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gjin10969/rvc-model-sample/resolve/main/suguirasan/suguirasan_voice.pth -d /root/jonathan_codev_tts/weights/suguirasan -o suguirasan_voice.pth

