import numpy as np
from PIL import Image
import pickle
import os
from numpy import loadtxt
import re # regular expressiion


# This functions create pckl files to each image in the all subfolders


def add_to_pickle(path, item):
    with open(path, 'wb') as file:
        pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data

def find_object_data_in_gt_file(image_name,gt_data):
    # find numbers in image_name
    tag_id=-1
    image_id=-1
    numbers_in_first_gt_row = np.array([int(s) for s in re.findall(r'-?\d+\.?\d*', gt_data[0])])
    rt=len(numbers_in_first_gt_row)
    gt_data_out=-1*np.array(np.ones([1,len(numbers_in_first_gt_row)]),dtype=int)
    num_in_image_name= np.array([int(s) for s in re.findall(r'-?\d+?\d*', image_name)])
    if len(num_in_image_name)==2:
        tag_id=num_in_image_name[0]
        image_id=num_in_image_name[1]
        # find all numbers in gt_line and compare obj_ind and image_ind

        for gt_row in gt_data:
            numbers_in_gt_row = np.array([int(s) for s in re.findall(r'-?\d+\.?\d*', gt_row)])
            if len(numbers_in_gt_row) > 2:
                if numbers_in_gt_row[0]==tag_id and numbers_in_gt_row[1]==image_id:
                    gt_data_out=numbers_in_gt_row
                    return tag_id,image_id,gt_data_out
    return tag_id,image_id,gt_data_out

def SaveImageDataToPckl(DirectoryToSave, ImageName,ObjectDataDict):
    Pklfilename = ImageName + '.dat'
    Pklfullfilename = os.path.join(DirectoryToSave, Pklfilename)
    add_to_pickle(Pklfullfilename, ObjectDataDict)

def CreatePcklFiles(DataDirectory):
    ######################################################
    valid_images = [".jpg", ".tif", ".png", ".tiff"]
    # Find ground_truth file in the directory
    gt_file = [f for f in os.listdir(DataDirectory) if f.endswith('.txt')]
    # Read gt_file
    gt_file_name=os.path.join(DataDirectory, gt_file[0])
    gt_file_fid = open(gt_file_name, "r")
    gt_file_lines = gt_file_fid.readlines()
    gt_lines_without_header=gt_file_lines[1:]
    gt_file_fid.close()
    #1. Find all subfolders in the current directory
    folders = [f.path for f in os.scandir(DataDirectory) if f.is_dir()]
    if len(folders) ==0:
        folders.append(DataDirectory)
    for curr_dir in folders:
        (head, ClassName) = os.path.split(curr_dir)
        file_list = [f for f in os.listdir(curr_dir)]
        io=len(file_list)
        # 2. Find all images in the current directory
        for idx,image in enumerate(file_list):
            #image='15036_img16490.png'
            print("Processing Image %d, from %d images %s" % (idx, io, image))
            ext = os.path.splitext(image)[1]
            if ext.lower() not in valid_images:
                continue
            ImageFileName = os.path.join(curr_dir, image)
            OneImage = load_image(ImageFileName)
            # 2. Find object data in ground truth file for the current image
            tag_id, image_id, gt_data_out=find_object_data_in_gt_file(image, gt_lines_without_header)
            if gt_data_out[0]==-1:# data not found in gt file
                ObjectDataValid = 0
            else:
                ObjectDataValid = 1
            if  gt_data_out[20]>7 or gt_data_out[20]<0:# color
                ty=1
            ObjectDataDict ={
                'ImageName' : image,
                'ImageVector' : np.ravel(OneImage),
                'ObjectDataValid' : ObjectDataValid,
                'ImageHeight' : OneImage.shape [0],
                'ImageWidth' : OneImage.shape [1],
                'ImageChannels' : OneImage.shape [2],
                'ObjectId' : tag_id,
                'ImageId' : image_id,
                'ObjectCenter_X' : gt_data_out[2],
                'ObjectCenter_Y' : gt_data_out[3],
                'ObjectLength' : gt_data_out[4],
                'ObjectWidth' : gt_data_out[5],
                'ObjectGeneral_class' : gt_data_out[6],
                'ObjectSub_class' : gt_data_out[7],
                'ObjectSunroof' : gt_data_out[8],
                'ObjectLuggage_carrier': gt_data_out[9],
                'ObjectOpen_cargo_area': gt_data_out[10],
                'ObjectEnclosed_cab': gt_data_out[11],
                'ObjectSpare_wheel': gt_data_out[12],
                'ObjectWrecked': gt_data_out[13],
                'ObjectFlatbed': gt_data_out[14],
                'ObjectLadder': gt_data_out[15],
                'ObjectEnclosed_box': gt_data_out[16],
                'ObjectSoft_shell_box': gt_data_out[17],
                'ObjectHarnessed_to_a_cart': gt_data_out[18],
                'ObjectAc_vents': gt_data_out[19],
                'ObjectColor': gt_data_out[20]}


            # gt_line structure:
            # tag_id  image_id  ObjectCenter_X  ObjectCenter_Y  ObjectLength  ObjectWidth  general_class  sub_class
            # sunroof  luggage_carrier  open_cargo_area  enclosed_cab  spare_wheel  wrecked  flatbed  ladder  enclosed_box
            # soft_shell_box  harnessed_to_a_cart  ac_vents  color

            # Save Object data to Pckl file
            SaveImageDataToPckl(curr_dir, image,ObjectDataDict)



## main
if __name__ == '__main__':
    TrainSetDir = 'C:\\Andrey\\DeepLearning\\MafatChallenge\\Data\\Train\\'
    CreatePcklFiles(TrainSetDir)
    #TestSetDir = 'C:\\Andrey\\DeepLearning\\MafatChallenge\\Data\\Test\\'
    #CreatePcklFiles(TestSetDir)
