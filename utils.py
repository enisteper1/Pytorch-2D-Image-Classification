import os
import cv2
import numpy as np
from tqdm import tqdm
import glob


class Create_Dataset:
    def __init__(self, im_size = 100, dataset = "dataset.npy", c_folder = "data/images", class_file = "data/classes.txt"):
        self.img_size = im_size
        self.dataset = dataset
        current_path = os.path.abspath(os.getcwd()).replace("\\","/")
        self.Main_Image_Path  = str(current_path) + "/" + c_folder + "/"
        self.Classes = list() #It will contain classes
        with open(class_file,"r") as c_file:
            self.Classes = c_file.read().split("\n")
        self.labels = dict((_class , i ) for i, _class in enumerate(self.Classes))
        self.one_hot_encoder = list()
        for i in range(len(self.Classes)):#creating one hot encoding arrays depending on class size
            encode = [0 for k in range(len(self.Classes))]
            encode[i] = 1
            self.one_hot_encoder.append(encode)
        self.training_data = list()
    def prepare_data(self):
        print("Creating Dataset")
        for i ,label in enumerate(self.labels):
            working_path = self.Main_Image_Path + str(label) + "/"
            extensions = ["*.jpg","*.jpeg", "*.png", "*.jpe", "*.tiff","*.exr","*.pfm","*.jp2","*.bmp","*.dib","*.pbm"]#accepted extension formats
            list_of_class = [glob.glob(working_path + x, recursive = True) for x in extensions]
            for type_list in list_of_class:
                for im_path in type_list:
                    try:
                        image = cv2.imread(im_path.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
                        image = cv2.resize(image, (self.img_size,self.img_size))
                        self.training_data.append([np.array(image),self.one_hot_encoder[i]])
                    except Exception as ex:
                        print(ex)
                        pass
            print(self.Classes[i])
        np.random.shuffle(self.training_data)
        np.save(self.dataset, self.training_data)





