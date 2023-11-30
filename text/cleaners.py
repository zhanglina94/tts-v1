import re
from text.korean import latin_to_hangul, number_to_hangul, divide_hangul, korean_to_lazy_ipa, korean_to_ipa, fix_g2pk2_error
from g2pk2 import G2p
from text.english import english_to_ipa, english_to_lazy_ipa, english_to_ipa2, english_to_lazy_ipa2
from unidecode import unidecode
from phonemizer import phonemize


_whitespace_re = re.compile(r'\s+')

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    # - For replication of https://github.com/FENRlR/MB-iSTFT-VITS2/issues/2
    # you may need to replace the symbol to Russian one
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = text.lower()
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
  return english_to_ipa(text)


def english_cleaners2(text):
  return english_to_ipa2(text)


def english_cleaners3(text): # needs espeak - apt-get install espeak
    text = convert_to_ascii(text)
    text = expand_abbreviations(text.lower())
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True,with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def korean_cleaners(text):
    '''Pipeline for Korean text'''
    text = latin_to_hangul(text)
    g2p = G2p()
    text = g2p(text)
    text = divide_hangul(text)
    text = fix_g2pk2_error(text)
    text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
    return text


def korean_cleaners2(text): # KO part from cjke
    '''Pipeline for Korean text'''
    korean_to_ipa(text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text









