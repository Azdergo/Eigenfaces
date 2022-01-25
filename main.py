import sys
import os
import math
import cv2 as cv
import numpy as np
from PIL import Image
import glob

recognizer = cv.face.LBPHFaceRecognizer_create()

PATH              = os.getcwd()
DATASET_NAME      = "dataset"
DATASET_PATH      = PATH + "/" + DATASET_NAME
BDD_NAME          = "bdd"
BDD_PATH          = PATH + "/" + BDD_NAME
ANALYSED_NAME     = "modified"
ANALYSED_PATH     = DATASET_PATH + "/" + ANALYSED_NAME
VECTOR_NAME       = "vector"
VECTOR_PATH       = DATASET_PATH + "/" + VECTOR_NAME
IMAGE_VECTOR_NAME = "image_vector.jpg"

DEST_WIDTH = 178
DEST_HEIGHT = 218
REF_ED = 39
REF_ELX = 69
REF_EY = 110

FACE_CASCADE = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE  = cv.CascadeClassifier('haarcascade_eye.xml')

class Image:
    def __init__(self, img, personId, fileName):
        self.img = img
        self.personId = personId
        self.fileName = fileName


def distance(p1, p2):
    return math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def readImages():
    image_list = []
    for root, subDirs in os.walk(BDD_PATH):
        for subDir in subDirs:
            imgPath = root + '/' + subDir + '/' + "*.jpg"
            for fileName in glob.glob(imgPath):
                img = cv.imread(fileName)
                image_list.append(Image(img, subDir, fileName))
    return image_list

def faceDetection(image_name):
    path = DATASET_PATH + "/" + image_name
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 6)
    for face in faces:
        (fx, fy, fw, fh) = face
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        le, re = eyesDetection(roi_gray, face)
        if le is not None and re is not None:
            image = Image.open(path)
            image = normalizeImage(image_name, le, re)
            image.save(ANALYSED_PATH + "/" + image_name)

def eyesDetection(image, face):
    (fx, fy, fw, fh) = face
    lec = rec = None
    eyes = EYE_CASCADE.detectMultiScale(image)
    if len(eyes) == 2:
        if eyes[0][0] < eyes[1][0]:
            (lex, ley, lew, leh) = eyes[0]
            (rex, rey, rew, reh) = eyes[1]
        else:
            (lex, ley, lew, leh) = eyes[1]
            (rex, rey, rew, reh) = eyes[0]
        lec = int((lex + (lew / 2)) + fx), int((ley + (leh / 2)) + fy)
        rec = int((rex + (rew / 2)) + fx), int((rey + (reh / 2)) + fy)
    return lec, rec

def rotateImage(image, el, er, resample=Image.BICUBIC):
    center = el
    direction = er[0] - el[0], er[1] - el[1]
    angle = -math.atan2(float(direction[1]), float(direction[0]))
    if center is None:
        return image.rotate(angle=angle, resample=resample)
    center_x, center_y = center
    scale_x = scale_y = 1.0
    cosinus = math.cos(angle)
    sinus = math.sin(angle)
    a = cosinus / scale_x
    b = sinus / scale_x
    c = center_x - (center_x * a) - (center_y * b)
    d = - sinus / scale_y
    e = cosinus / scale_y
    f = center_y - (center_x * d) - (center_y * e)
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

def scaleImage(image):
    return image.resize((DEST_WIDTH, DEST_HEIGHT), Image.ANTIALIAS)

def cropImage(image, el, er, avg_color):
    ed = distance (el, er)
    scale = float(ed) / float(REF_ED)
    crop_ul = el[0] - int(scale * REF_ELX), el[1] - int(scale * REF_EY)
    crop_size = int(DEST_WIDTH * scale), int(DEST_HEIGHT * scale)
    crop_dr = crop_ul[0] + crop_size[0], crop_ul[1] + crop_size[1]

    return image.crop((crop_ul[0], crop_ul[1], crop_dr[0], crop_dr[1]))

def normalizeImage(image_name, el, er):
    image = Image.open(DATASET_PATH + "/" + image_name)
    avg_color_per_channel = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_channel, axis=0)
    
    image = rotateImage(image, el, er)
    image = cropImage(image, el, er, avg_color)
    image = scaleImage(image)
    return image

def concatImage(actual=None, new=None):
    if actual is None and new is None:
        return None
    if new is None:
        return actual
    new = cv.imread(new)
    if actual is None:
        return new
    return cv.hconcat([actual, new])

def getAllFaces(image_list):
    faces = []
    i = 0
    for image in image_list:
        i += 1
        faceImg = normalizeImage(image)
        if faceImg is not None:
            faces.append(faceImg)
    return faces

def train():
    # Read images
    print("\t Lecture des images...")
    images = readImages()

    print("\t Reconnaissance des visages...")
    imgWithfaces = getAllFaces(images)
    faces = []
    labels = []
    i = 0
    for image in imgWithfaces:
        faces.append(image.img)
        labels.append(i)
        i += 1

    print("\t Entraînement du modèle avec les images...")
    recognizer.train(faces, np.array(labels))

# if __name__ == "__main__":