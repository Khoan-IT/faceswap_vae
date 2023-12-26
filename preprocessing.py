import cv2
import glob
import os
import argparse
from torch_snippets import *

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img):
    return cv2.resize(img, (256, 256)), True


def crop_images(folder):
    images = glob.glob(folder + '/*.jpg')
    for i in range(len(images)):
        people = images[i].split('/')[-2]
        img = read(images[i], 1)
        img2, face_detected = crop_face(img)
        cropped_folder = folder.replace(people, 'cropped_faces_' + people)
        if not os.path.isdir(cropped_folder):
            os.makedirs(cropped_folder, exist_ok=True)

        if(face_detected == False):
            continue
        else:
            cv2.imwrite(cropped_folder + '/' + str(i) + '.jpg', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Code preprocess data')
    parser.add_argument("--input_folder", required=True, type=str, help="Path to folder consisting images")
    
    args = parser.parse_args()
    
    crop_images(args.input_folder)
    
    # Usage example: python preprocessing.py --input_folder=./data/hai
    # The result of images will save in ./data/cropped_faces_hai     
    