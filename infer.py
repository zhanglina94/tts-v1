import librosa
import matplotlib.pyplot as plt

import os
import json
import math

import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
from models import SynthesizerTrn
from text import text_to_sequence
from scipy.io.wavfile import write
import re
from scipy import signal
import time


# - paths
path_to_config = "./config.json" # path to .json
path_to_model = "./best.pth" # path to G_xxxx.pth



#- text input
input = "소프트웨어 교육의 중요성이 날로 더해가는데 학생들은 소프트웨어 관련 교육을 쉽게 지루해해요."

# check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"

_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) # !

SPACE_ID = symbols.index(" ")


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def vcss(inputstr): # single
    print('text:',inputstr)
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    stn_tst = get_text(fltstr, hps)
    speed = 1
    output_dir = 'output'
    sid = 0
    start_time=time.time()
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
                0, 0].data.cpu().float().numpy()
    write(f'./{output_dir}/tts_output.wav', hps.data.sampling_rate, audio)
    print(f'./{output_dir}/output file Generated!')
    end_time=time.time()
    runTime=end_time-start_time
    print("RunTime:{}sec".format(runTime))


def vcms(inputstr, sid): # multi
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    #fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
    stn_tst = get_text(fltstr, hps)

    speed = 1
    output_dir = 'output'
    start_time=time.time()
    with torch.no_grad():
        
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([sid]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
            0, 0].data.cpu().float().numpy()
    write(f'./{output_dir}/output.wav', hps.data.sampling_rate, audio)
    end_time=time.time()
    print(f'./{output_dir}/output file Generated!')
    end_time=time.time()
    runTime=end_time-start_time
    print("RunTime:{}sec".format(runTime))

hps = utils.get_hparams_from_file(path_to_config)

if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers, #- >0 for multi speaker
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(path_to_model, net_g, None)


vcss(input)



