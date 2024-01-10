import pandas as pd
import csv

# csv2txt
wav_dir="/Path/to/wavs/"
new_text_file=open("/Path/to/new.txt",'w')
filename = '/Path/to/original.csv'


with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row)
        if row['correct']=='O':
            new_text_file.write(wav_dir+ row['row_no']+".wav"+'|2|'+row["new_text"]+'\n') 

# txt2csv
def txt_data_to_csv(path,save_dir):

    files= os.listdir(path)
    for txt in files: 
        file_path = os.path.join(path,'transcript.txt')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path=os.path.join(save_dir,'transcript.csv')
        if os.path.exists(save_path):
            os.remove(save_path)  
        x_list=[]
        y_list=[]
        with open(file_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                line = line.lstrip()
                print(line)
                if line!=' ':
                    xy=line.split()
                    x_list.append(xy[0]+'.wav|')
                    y_list.append(xy[1:-1])
    
            rows = zip(x_list,y_list)
            with open(save_path, "w", newline='') as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)

path='/path/to/data'
save_dir='/path/to/new_data'
txt_data_to_csv(path,save_dir)
