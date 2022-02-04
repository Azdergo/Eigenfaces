import sys
import os
import math
import cv2
import numpy as np
from PIL import Image
import glob
from copy import copy
import transform_image


recognizer = cv2.face.LBPHFaceRecognizer_create()

PATH              = os.getcwd()
DATASET_NAME      = "dataset"
DATASET_PATH      = PATH + "/" + DATASET_NAME
ANALYSED_NAME     = "modified"
ANALYSED_PATH     = DATASET_PATH + "/" + ANALYSED_NAME
BDD_NAME          = "bdd"
BDD_PATH          = PATH + "/" + BDD_NAME
NB_IMAGE_PER_ID   = 30

persons = []

class MyImage:
    def __init__(self, img, personId, fileName):
        self.img = img
        self.personId = personId
        self.fileName = fileName


def getAllFaces(image_list):
    faces = []
    i = 0
    for image in image_list:
        i += 1
        image, face = transform_image.detectFace(image)
        if face is not None:
            faces.append(image)
    return faces

def train():
    # Read images
    images = readImages()

    print("\t Reconnaissance des visages...")
    imgWithfaces = getAllFaces(images)
    faces = labels = []
    i = 0
    for image in imgWithfaces:
        faces.append(image.img)
        persons.append(image.personId)
        labels.append(i)
        i += 1
    recognizer.train(faces, np.array(labels))

def prediction(input):
    image = copy(input)
    imageAfterDetection, face = transform_image.detectFace(image)
    if face is None:
        return None
    label, confidence = recognizer.predict(image.img)
    name = persons[label]
    image.img = cv2.imread(BDD_PATH + image.personId + image.fileName)
    return image, name, confidence

def getImagesList(id_path):
    return [f for f in os.listdir(id_path) if os.path.isfile(os.path.join(id_path, f))]

def getIdDirs():
    return [d for d in os.listdir(BDD_PATH)]

def readImages(path):
    image_list = []
    for root, subdirectories, files in os.walk(path):
        for subdirectory in subdirectories:
            imgPath = root + '/' + subdirectory + '/' + "*.jpg"
            for filename in glob.glob(imgPath):
                img = cv2.imread(filename)
                image_list.append(Image(img, subdirectory, filename))
    return image_list

def main(path):
    train()
    pathNormalizedImage = transform_image.faceDetection(path)
    img = cv2.imread(pathNormalizedImage)
    
    splitted_path = pathNormalizedImage.split("/")
    fileName = splitted_path[-1][:-4]
    dirName = splitted_path[-2]
    image = Image(img, dirName, fileName)
    prediction, confidence = prediction(image)
    print("Confidence: ", confidence)
    cv2.imshow("Resultat", cv2.resize(prediction, (400, 500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ids = getIdDirs()
    for id in ids:
        print(getImagesList(id)[:int(NB_IMAGE_PER_ID*0.7)])

