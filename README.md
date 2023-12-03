# Text-to-speech-ko

### Env 
```
git clone https://github.com/zhanglina94/tts-v1
pip install -r requirements.txt
apt-get install espeak

# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace

```


### Download pre-trained weight and config file
```sh
python infer.py
```


## Credits
- [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [misakiudon/MB-iSTFT-VITS-multilingual](https://github.com/misakiudon/MB-iSTFT-VITS-multilingual)
- [FENRlR/MB-iSTFT-VITS2](https://github.com/FENRlR/MB-iSTFT-VITS2)
