import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def FindAllPklFilesInDirectory(DirectoryName,extension):
    #files_pkl = (f for f in os.listdir(DirectoryName) if f.endswith('.' + extension))
    files = os.listdir(DirectoryName)
    files_pkl = [i for i in files if i.endswith('.' + extension)]
    return files_pkl

def ReadPklFile(PklFileName):
    with open(PklFileName, 'rb') as f:
        x= pickle.load(f)
    return x

def ReadAllPklFile(PathName):
    all_files_data = []
    folders = [f.path for f in os.scandir(PathName) if f.is_dir()]
    if len(folders) == 0:
        folders.append(PathName)
    for cur_dir in folders:
        full_curr_dir_name = os.path.join(PathName, cur_dir)
        filesList = FindAllPklFilesInDirectory(full_curr_dir_name, 'dat')
        for pklfile in filesList:
            filename = os.path.join(full_curr_dir_name, pklfile)
            fileData = ReadPklFile(filename)
            all_files_data.append(fileData)
    return all_files_data





if __name__ == '__main__':
    PathName='C:\\Andrey\\DeepLearning\\TensorF\\PointTargetProject\\MydataDir\\'
    allPcklFilesData=ReadAllPklFile(PathName)
    ty=1
