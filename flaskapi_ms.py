
import sys
from flask import Flask, request, jsonify
from flask.views import MethodView
from flask_cors import CORS
import argparse
import base64

import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import logging

import soundfile
import torch

import commons
import wave
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import re
from scipy import signal
import time

# Global variables
device = "cuda:0" if torch.cuda.is_available() else "cpu"
net_g = None
hps = None

app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})

class run_api(MethodView):
    @api_monitor()
    def get(self):
        return "Test"

    @api_monitor()
    def post(self):
        jf_text1 = request.json['personID']
        jf_text2 = request.json['text']
        
        # Load model based on personID
        load_model(jf_text1)

        def get_text(text, hps):
            text_norm = text_to_sequence(text, hps.data.text_cleaners)
            if hps.data.add_blank:
                text_norm = commons.intersperse(text_norm, 0)
            text_norm = torch.LongTensor(text_norm)
            return text_norm

        fltstr = re.sub(r"[\[\]\(\)\{\}]", "", jf_text2)
        stn_tst = get_text(fltstr, hps)
        speed = 1

        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
                    0, 0].data.cpu().float().numpy()
            print("audio type:", audio, audio.shape)

        audio_file = write('./output/out.wav', hps.data.sampling_rate, audio)
        out_path = "./output/out.wav"

        if os.path.exists(out_path):
            output = {
                "audio": [
                    {
                        "@TITLE": base64.b64encode(open(out_path, 'rb').read()).decode()
                    }
                ]
            }
            print("output:", output)
            return jsonify(output)
        else:
            return send_file(out_path, mimetype="audio/wav", as_attachment=True, download_name="out.wav")


def load_model(personID):
    global net_g, hps
    path_to_config = "./config.json"

    if personID == '0':
        path_to_model = "./G_0.pth"
    elif personID == '1':
        path_to_model = "./G_1.pth"
    elif personID == '2':
        path_to_model = "./G_2.pth"
    else:
        path_to_model = "./G_0.pth"

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
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(path_to_model, net_g, None)


app.add_url_rule("/", view_func=run_api.as_view("run_api"))

if __name__ == "__main__":
    app.run('0.0.0.0', 8888, threaded=True)
