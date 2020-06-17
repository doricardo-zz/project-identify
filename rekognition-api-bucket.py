import os
import csv
import boto3
import pandas as pd
import time
from tqdm import tqdm

confidence = 90

with open('credentials.csv','r') as input:
    next(input)
    reader = csv.reader(input)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

client = boto3.client('rekognition',
                      region_name='us-east-1',
                      aws_access_key_id = access_key_id,
                      aws_secret_access_key = secret_access_key)

#bucket
s3 = boto3.resource('s3',
                    region_name='us-east-1',
                    aws_access_key_id = access_key_id,
                    aws_secret_access_key = secret_access_key)

bucket = s3.Bucket('project-identify')
path = 'upload/'

photos = []
for file in bucket.objects.all():
    photos.append(file.key)

file = []
for photo in tqdm(photos):

    response = client.detect_text(Image={'S3Object': { 'Bucket': 'project-identify', 'Name': photo} }, Filters={'WordFilter': {'MinConfidence': confidence} })

    textDetections = response['TextDetections']
    bibs = []
    bibs.append(photo)
    for text in textDetections:
        if text['DetectedText'].isnumeric() and text['Type'] == 'LINE':
            bibs.append(text['DetectedText'])

    file.append(bibs)

df = pd.DataFrame(file)

df.to_csv('output.csv', index=False, header=False, sep=';')

#------------------------------------------------------
col = ['file']
for i in range(df.columns.stop-1):
    col.append('bib'+str(i+1))

df.columns = col
print("Percentual identificado")
print(round((df.bib1.count() / df.file.count()) * 100))
#------------------------------------------------------