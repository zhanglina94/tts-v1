# Text-to-speech-ko


### Only Test~

```sh
git clone https://github.com/zhanglina94/tts-v1
cd tts-v1
```

### Download pre-trained weight and config file


### Env Setting

Install 
```
pip install -r requirements.txt
apt-get install espeak

# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace

```

### Download pre-trained weight and config file

### Inference
```sh
python infer.py
```


### Credits
- [FENRlR/MB-iSTFT-VITS2](https://github.com/FENRlR/MB-iSTFT-VITS2)
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [ORI-Muchim/PolyLangVITS](https://github.com/ORI-Muchim/PolyLangVITS)
- [misakiudon/MB-iSTFT-VITS-multilingual](https://github.com/misakiudon/MB-iSTFT-VITS-multilingual)
- [innnky/emotional-vits](https://github.com/innnky/emotional-vits/tree/main)
