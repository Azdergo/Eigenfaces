import os
import math
import cv2 as cv
import numpy as np
from PIL import Image

PATH          = os.getcwd()
DATASET_NAME  = "dataset"
DATASET_PATH  = PATH + "/" + DATASET_NAME
ANALYSED_NAME = "modified"
ANALYSED_PATH = DATASET_PATH + "/" + ANALYSED_NAME

DEST_WIDTH = 178
DEST_HEIGHT = 218
REF_ED = 39
REF_ELX = 69
REF_EY = 110

FACE_CASCADE = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE  = cv.CascadeClassifier('haarcascade_eye.xml')

def distance(p1, p2):
    return math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def init_result_folder():
    if not os.path.exists(ANALYSED_PATH):
        os.makedirs(ANALYSED_PATH)
    

def get_images_list():
    return [f for f in os.listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, f))]

def face_detection(image_name):
    path = DATASET_PATH + "/" + image_name
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 6)
    for face in faces:
        (fx, fy, fw, fh) = face
        roi_color = img[fy:fy + fh, fx:fx + fw]
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        le, re = eyes_detection(roi_gray, face)
        if le is not None and re is not None:
            imgage = Image.open(path)
            imgage = normalize_image(image_name, le, re)
            imgage.save(ANALYSED_PATH + "/" + image_name)

def eyes_detection(image, face):
    (fx, fy, fw, fh) = face
    lec = rec = None
    eyes = EYE_CASCADE.detectMultiScale(image)
    if len(eyes) == 2:
        if eyes[0][0] < eyes[1][0]:
            (lex, ley, lew, leh) = le = eyes[0]
            (rex, rey, rew, reh) = re = eyes[1]
        else:
            (lex, ley, lew, leh) = le = eyes[1]
            (rex, rey, rew, reh) = re = eyes[0]
        lec = int((lex + (lew / 2)) + fx), int((ley + (leh / 2)) + fy)
        rec = int((rex + (rew / 2)) + fx), int((rey + (reh / 2)) + fy)
    return lec, rec

def rotate_image(image, el, er, resample=Image.BICUBIC):
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

def scale_image(image):
    return image.resize((DEST_WIDTH, DEST_HEIGHT), Image.ANTIALIAS)

def crop_image(image, el, er, avg_color):
    ed = distance (el, er)
    scale = float(ed) / float(REF_ED)
    crop_ul = el[0] - int(scale * REF_ELX), el[1] - int(scale * REF_EY)
    crop_size = int(DEST_WIDTH * scale), int(DEST_HEIGHT * scale)
    crop_dr = crop_ul[0] + crop_size[0], crop_ul[1] + crop_size[1]

    return image.crop((crop_ul[0], crop_ul[1], crop_dr[0], crop_dr[1]))

    if crop_ul[0] < 0:
        image = cv.hconcat([blank_image, image])
        blank_image = cv.hconcat([blank_image, blank_image])
        print("coucou 1")
        crop_ul[0] += width
        crop_dr[0] += width
    if crop_dr[0] > width:
        image = cv.hconcat([image, blank_image])
        blank_image = cv.hconcat([blank_image, blank_image])
        print("coucou 2")
    if crop_ul[1] < 0:
        image = cv.vconcat([blank_image, image])
        blank_image = cv.vconcat([blank_image, blank_image])
        print("coucou 3")
        crop_ul[1] += height
        crop_dr[1] += height
    if crop_dr[1] > height:
        image = cv.vconcat([image, blank_image])
        blank_image = cv.vconcat([blank_image, blank_image])
        print("coucou 4")

    return image[
        crop_ul[1]:crop_dr[1],
        crop_ul[0]:crop_dr[0]
    ]

def normalize_image(image_name, el, er):
    image = Image.open(DATASET_PATH + "/" + image_name)
    avg_color_per_channel = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_channel, axis=0)
    
    image = rotate_image(image, el, er)
    image = crop_image(image, el, er, avg_color)
    image = scale_image(image)
    return image


if __name__ == "__main__":
    init_result_folder()
    for img_name in get_images_list():
        face_detection(img_name)
