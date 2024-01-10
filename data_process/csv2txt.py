import pandas as pd
import csv

wav_dir="/Path/to/wavs/"
new_text_file=open("/Path/to/new.txt",'w')
filename = '/Path/to/original.csv'


with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row)
        if row['correct']=='O':
            new_text_file.write(wav_dir+ row['row_no']+".wav"+'|2|'+row["new_text"]+'\n') 

