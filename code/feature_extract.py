from __future__ import print_function
import os  # needed navigate the system to get the input data
import pandas as pd
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import csv

dataDir = '...'
folder_List = os.listdir(dataDir)

yaml_file = r'./parameter.yaml'
out_path = r'...'

extractor = featureextractor.RadiomicsFeatureExtractor(yaml_file)
namelist = []
valuelist = []
result = []

for folder in folder_List:
    imageName = dataDir + folder + '/' + folder + '.nrrd'
    maskName = dataDir + folder + '/' + folder + '_mask.nrrd'
    print(imageName)
    print(maskName)
    result = extractor.execute(imageName, maskName)
    valuelist.append(result.values())
    namelist.append(folder)

with open("./csv.csv",'w',newline='') as t:
    writer = csv.writer(t)
    writer.writerow(result.keys())
    writer.writerows(valuelist)
csv = pd.read_csv("./csv.csv")

features = csv.iloc[:, 37:]
features.insert(0, 'name', namelist)

features.to_csv(out_path, index=False)