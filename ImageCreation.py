import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pickle
import os

## This functions creates 28x28 images wih gaussian noise as a background
## Point Target is inserted to each image in the random pixel

def CreateOneImage(ImageDC,BackgSigma,ObjectSNR,XObject,YObject,AddObject):
    # Create Background
    ImageWidth=28
    ImageHeight=28
    ImageBackg=ImageDC+np.random.randn(ImageWidth,ImageHeight)*BackgSigma
    OneImage=ImageBackg
    # Add Object
    if(AddObject):
        PSF_Filter=np.array([[0.1224, 0.3061, 0.1227],
                    [0.3061, 1, 0.3061],
                    [0.1224, 0.3061, 0.1227]])
        ObjectPeak=BackgSigma*ObjectSNR
        ObjectPatch=ObjectPeak*PSF_Filter+ImageDC
        OneImage[YObject-1:YObject+2,XObject-1:XObject+2]=ObjectPatch
    return OneImage

def add_to_pickle(path, item):
    with open(path, 'wb') as file:
        pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)

class ImageObject(object):
    def __init__(self, ImageVector,ImageLabel,ImageName,ObjectDataValid,ObjectXLoc,ObjectYLoc,ObjectSNR):
        self.ImageVector = ImageVector
        self.ImageLabel = ImageLabel
        self.ImageName = ImageName
        self.ObjectDataValid = ObjectDataValid
        self.ObjectXLoc = ObjectXLoc
        self.ObjectYLoc = ObjectYLoc
        self.ObjectSNR = ObjectSNR

    def SaveImageObjectToPckl(self,DirectoryToSave):
        Pklfilename=self.ImageName+'.dat'
        Pklfullfilename=os.path.join(DirectoryToSave, Pklfilename)
        add_to_pickle(Pklfullfilename, self.ImageVector)
        add_to_pickle(Pklfullfilename, self.ImageLabel)
        add_to_pickle(Pklfullfilename, self.ImageName)
        add_to_pickle(Pklfullfilename, self.ObjectDataValid)
        add_to_pickle(Pklfullfilename, self.ObjectXLoc)
        add_to_pickle(Pklfullfilename, self.ObjectYLoc)
        add_to_pickle(Pklfullfilename, self.ObjectSNR)


def SaveImageDataToPckl(DirectoryToSave,ImageVector,ImageLabel,ImageName,ObjectDataValid,ObjectXLoc,ObjectYLoc,ObjectSNR):
    Pklfilename = ImageName + '.dat'
    Pklfullfilename = os.path.join(DirectoryToSave, Pklfilename)
    ImageDataDict={'ImageVector' : ImageVector,
                   'ImageLabel' : ImageLabel,
                   'ImageName' : ImageName,
                   'ObjectDataValid' : ObjectDataValid,
                   'ObjectXLoc' : ObjectXLoc,
                   'ObjectYLoc' : ObjectYLoc,
                   'ObjectSNR' : ObjectSNR}
    add_to_pickle(Pklfullfilename, ImageDataDict)



## main

NumOfImagesWithObjectToCreate=100
ImageDC=46000
BackgSigma=10
ObjectSNR=4
ImageWidth=28
ImageHeight=28
ImageFormatToSave='.png'
XObject=np.random.randint(low=1, high=ImageWidth-1, size=NumOfImagesWithObjectToCreate)
YObject=np.random.randint(low=1, high=ImageHeight-1, size=NumOfImagesWithObjectToCreate)
ObjectSNR=np.random.randint(low=4, high=10, size=NumOfImagesWithObjectToCreate)
current_directory=os.getcwd()


######################################################
######## Object data creation ########################
Object_directory=os.path.join(current_directory, "PointTargetProject\\Object")
if not os.path.exists(Object_directory):
        os.makedirs(Object_directory)
# Object ground truth file definition
gt_file_name="ground_truth.txt"
full_gt_file_name=os.path.join(Object_directory, gt_file_name)
LinesToSaveInObjectGT=[]
Header="FrameNumber \tDataValid  \tXloc\tYloc\tSNR\t     ImageName\n" # \t = a tab
LinesToSaveInObjectGT.append(Header)
## Create images loop
print("Create Object data")
for i in range(NumOfImagesWithObjectToCreate):
    AddObject=True
    imageNum=i
    Xc = XObject[i]
    Yc = YObject[i]
    print("Image num %d XObj %d YObj %d" %(imageNum,Xc,Yc))
    OneImage=CreateOneImage(ImageDC,BackgSigma,ObjectSNR[i],Xc,Yc,AddObject)
    # Save the image
    ImageName= "img%06d" % (imageNum)
    Imagefilename = ImageName+ImageFormatToSave
    Fullfilename=os.path.join(Object_directory, Imagefilename)
    #im = Image.fromarray(OneImage)
    ar32 = OneImage.astype(np.uint32)
    im = Image.fromarray(ar32)  # or more verbose as Image.fromarray(ar32, 'I')
    im.save(Fullfilename)
    # ["FrameNumber\tDataValid\tXloc\tYloc\tSNR\tImageName"]
    GTLine="  %04d\t           1\t     %d\t     %d\t     %d\t  %s\n" %(imageNum,Xc,Yc,ObjectSNR[i],Imagefilename)
    LinesToSaveInObjectGT.append(GTLine)
    #plt.imshow(OneImage, cmap='gray')
    #plt.show()

    ## Create Image Object
    ImageVector=np.ravel(OneImage)
    ImageLabel=1
    #ImageObj=ImageObject(ImageVector,ImageLabel,ImageName,1,Xc,Yc,ObjectSNR[i])
    #ImageObj.SaveImageObjectToPckl(Object_directory)
    SaveImageDataToPckl(Object_directory, ImageVector, ImageLabel, ImageName, 1, Xc, Yc, ObjectSNR[i])
    PklFileName=os.path.join(Object_directory,ImageName+'.dat')
    with open(PklFileName, 'rb') as f:
        x = pickle.load(f)
    ty=1

# Write ground_truth_file
with open(full_gt_file_name, 'w') as gt_file:
    gt_file.writelines(LinesToSaveInObjectGT)

######################################################
######## Object data creation ########################
# NoObject ground truth file definition
WithoutObject_directory=os.path.join(current_directory, "PointTargetProject\\NoObject")
if not os.path.exists(WithoutObject_directory):
        os.makedirs(WithoutObject_directory)

NoObj_gt_file_name="ground_truth.txt"
full_NoObj_gt_file_name=os.path.join(WithoutObject_directory, NoObj_gt_file_name)
LinesToSaveInNoObjGT=[]
LinesToSaveInNoObjGT.append(Header)

NumOfImagesNoObjectToCreate=NumOfImagesWithObjectToCreate

## Create images loop
print("Create NOObject data")
for i in range(NumOfImagesNoObjectToCreate):
    AddObject=False
    imageNum=i
    Xc = -1
    Yc = -1
    SNR = -1
    print("Image num %d XObj %d YObj %d" % (imageNum, Xc, Yc))
    OneImage = CreateOneImage(ImageDC, BackgSigma, SNR,Xc, Yc, AddObject)
    # Save the image
    ImageName = "img%06d" % (imageNum)
    Imagefilename = ImageName + ImageFormatToSave
    Fullfilename = os.path.join(WithoutObject_directory, Imagefilename)
    #im = Image.fromarray(OneImage)
    ar32 = OneImage.astype(np.uint32)
    im = Image.fromarray(ar32)  # or more verbose as Image.fromarray(ar32, 'I')
    im.save(Fullfilename)
    # ["FrameNumber\tDataValid\tXloc\tYloc\tSNR\tImageName"]
    GTLine = "  %04d\t           0\t     %d\t     %d\t     %d\t  %s\n" % (imageNum, Xc, Yc, SNR, Imagefilename)
    LinesToSaveInNoObjGT.append(GTLine)

    ## Create Image Object
    ImageVector = np.ravel(OneImage)
    ImageLabel = 0
    ImageObj = ImageObject(ImageVector, ImageLabel, ImageName, 0, Xc, Yc, SNR)
    ImageObj.SaveImageObjectToPckl(WithoutObject_directory)
# Write ground_truth_file
with open(full_NoObj_gt_file_name, 'w') as gt_file:
    gt_file.writelines(LinesToSaveInNoObjGT)