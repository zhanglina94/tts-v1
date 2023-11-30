
import sys
from flask import Flask, request, jsonify,render_template
from flask.views import MethodView
from flask_cors import CORS
import argparse
import base64
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import logging

import soundfile
import torch

from flask import Flask, request, send_file,jsonify
from flask_cors import CORS
from flask.views import MethodView
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import re
from scipy import signal
import time


# check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})


@app.route('/', methods=['GET','POST'])
def test():
    text = request.form["text"]
    print('text:', text)

    
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", text)
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
    output = write(f'./{output_dir}/out.wav', hps.data.sampling_rate, audio)
    out_path = "./output/out.wav"

    return send_file(out_path,mimetype="audio/wav", as_attachment=True,download_name="out.wav")



if __name__ == '__main__':

    path_to_config = "config.json" 
    path_to_model = "best.pth"
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

    app.run(port=6842, host="0.0.0.0", debug=True, threaded=False)
