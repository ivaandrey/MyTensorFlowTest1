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





if __name__ == '__main__':
    PathName='C:\\Andrey\\DeepLearning\\TensorF\\PointTargetProject\\Object'
    filesList=FindAllPklFilesInDirectory(PathName, 'dat')
    filename=os.path.join(PathName,filesList[0])
    fileData=ReadPklFile(filename)
    ImageVector=fileData['ImageVector']
    print(fileData)
    ty=1