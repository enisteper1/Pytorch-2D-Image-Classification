import glob
import argparse
import os
from model import *
import cv2
import numpy as np

class Detection:
    def __init__(self,args):
        self.device, self.img_size, self.inp_folder, self.class_file = args.device, args.img_size, args.inp_folder, args.class_file
        if self.device == "":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("Using GPU...")
            else:
                self.device = torch.device("cpu")
                print("Using CPU...")
    def detect(self, save_txt = False):
        with open(self.class_file, "r") as c_file:
            self.Classes = c_file.read().split("\n")
        print("Classes:\n\t", self.Classes)
        self.labels = dict((_class, i) for i, _class in enumerate(self.Classes))
        self.net = Net(nc = len(self.Classes), img_size = self.img_size).to(self.device)
        self.net.load_state_dict(torch.load("models/best.pth.tar"))
        output_path = os.path.abspath(os.getcwd()).replace("\\", "/") + "/data/outputs/"
        working_path = os.path.abspath(os.getcwd()).replace("\\", "/") + "/" + self.inp_folder + "/"
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.jpe", "*.tiff", "*.exr", "*.pfm", "*.jp2", "*.bmp", "*.dib",
                      "*.pbm"]
        inp_list = [glob.glob(working_path + x, recursive=True) for x in extensions]
        for type_list in inp_list:
            for i,im_path in enumerate(type_list):
                try:
                    im_path = im_path.replace("\\","/")
                    org_image = cv2.imread(im_path)
                    image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (self.img_size, self.img_size))
                    img_arr = torch.Tensor(np.array(image)) / 255.0
                    img_arr = img_arr.view(-1, 1, self.img_size, self.img_size).to(self.device)
                    prediction = self.Classes[torch.argmax(self.net(img_arr))]
                    org_image = cv2.putText(org_image, str(prediction), (int(0.1 * org_image.shape[0]), int(0.2 * org_image.shape[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imwrite(output_path + im_path.split("/")[-1], org_image)

                    if save_txt:
                        with open(output_path + im_path.split("/")[-1].split(".")[0] + ".txt", "w") as f:
                            f.write(str(prediction))
                except Exception as ex:
                    print(ex)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="", help="device_id")
    parser.add_argument("--img_size", type=int, default=64, help="Image sizes of input images")
    parser.add_argument("--inp_folder", type=str, default="data/inputs",help="Folder that contains input images")
    parser.add_argument("--class_file", type=str, default="data/classes.txt", help="File that contains your classes")
    parser.add_argument("--save_txt", action="store_true", help="Save prediction to txt")
    args = parser.parse_args()

    with torch.no_grad():
        detection = Detection(args)
        detection.detect(args.save_txt)