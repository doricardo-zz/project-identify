import os
import csv
import boto3
import pandas as pd
import time
from tqdm import tqdm

confidence = 95
path = "upload/"

def convert(photo):
    with open(path + photo,'rb') as source_image:
        source_bytes = source_image.read()
    return source_bytes

with open('credentials.csv','r') as input:
    next(input)
    reader = csv.reader(input)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

client = boto3.client('rekognition',
                      #region_name='us-east-1',
                      region_name='ap-northeast-2',
                      #region_name='us-west-2',
                      aws_access_key_id = access_key_id,
                      aws_secret_access_key = secret_access_key)

t0 = time.time()

photos = [f for f in os.listdir(path) if (f.lower().endswith(".jpg") or f.lower().endswith(".png"))]
df_dir = pd.DataFrame(photos, columns=['file'])
df_dir['bytes'] = df_dir['file'].apply(convert)
df_dir.set_index('file', inplace=True)

file = []
for photo in tqdm(photos):
    
    source_bytes = df_dir.loc[photo].bytes

    response = client.detect_text(Image={'Bytes':source_bytes}, Filters={'WordFilter': {'MinConfidence': confidence} })
    textDetections = response['TextDetections']
    bibs = []
    bibs.append(photo)
    for text in textDetections:
        if text['DetectedText'].isnumeric() and text['Type'] == 'LINE':
            bibs.append(text['DetectedText'])

    file.append(bibs)

df = pd.DataFrame(file)

df.to_csv('output.csv', index=False, header=False, sep=';')
t1 = time.time()

#-----------------------------------------------------------
col = ['file']
for i in range(df.columns.stop-1):
    col.append('bib'+str(i+1))

df.columns = col

print("Elpased time")
total = t1-t0
print(total/60)
print("Percentage identified")
print(round((df.bib1.count() / df.file.count()) * 100))
#-----------------------------------------------------------