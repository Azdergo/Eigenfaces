import cv2
import numpy as np
import math

# Importing HARR CASCADE XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

base = 'test'
img_name = 'benjamin'
# Uploading test image
img = base + '/' + img_name + '.jpg'
img = cv2.imread(img)

avg_color_per_row = np.average(img, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)

# Displaying the image
cv2.imshow('Detected Face Image', img)

# Waiting for escape key for image to close
cv2.waitKey()

height, width, channels = img.shape
center = (width / 2, height / 2)
print("Largeur:", height, ", Longueur:", width, "De l'image")

# Valeur des images de celebA
eyesGap = 39
eyesLargeur = 89.0
eyesLongueur = 109.0
dimensions = (178, 218)

# Converting to grey scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def distance(xa, ya, xb, yb):
    return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)


def rotate_image(image2, angle):
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(src=image2, M=rotate_matrix, dsize=(width, height), borderValue=avg_color)
    cv2.waitKey(0)

    return rotated_image


def crop(image, fCenter, rapport):
    # x,y du point en haut à gauche de la nouvelle image
    xa = int(fCenter[0] - (eyesLargeur * rapport))
    ya = int(fCenter[1] - (eyesLongueur * rapport))

    # x,y du point en bas à droite de la nouvelle image
    xb = int(fCenter[0] + (eyesLargeur * rapport))
    yb = int(fCenter[1] + (eyesLongueur * rapport))
    crop_img = image

    blank_image = cv2.imread('test/' + img_name + '.jpg')
    blank_image[:] = avg_color

    if xa < 0:
        crop_img = cv2.hconcat([blank_image, crop_img])
        blank_image = cv2.hconcat([blank_image, blank_image])
        xa = width + xa
        xb = width + xb

    if ya < 0:
        crop_img = cv2.vconcat([blank_image, crop_img])
        blank_image = cv2.vconcat([blank_image, blank_image])
        ya = height + ya
        yb = height + yb

    if xb > width:
        crop_img = cv2.hconcat([crop_img, blank_image])
        blank_image = cv2.hconcat([blank_image, blank_image])

    if yb > height:
        crop_img = cv2.vconcat([crop_img, crop_img])
        blank_image = cv2.vconcat([blank_image, blank_image])

    # On rogne l'image
    print(xa, ya, xb, yb)
    crop_img = crop_img[ya:yb, xa:xb]
    return crop_img


def resize(image2):
    resize_image = cv2.resize(image2, dimensions, interpolation=cv2.INTER_LINEAR)
    return resize_image


# Allowing multiple scale(Multiple size) detection
faces = face_cascade.detectMultiScale(gray, 1.1, 6)

# Creating Rectangle around face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 250), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    img_vierge = 'test/' + img_name + '.jpg'
    img_vierge = cv2.imread(img_vierge)

    if len(eyes) > 0:
        xle = int(eyes[0][0]) + (int(eyes[0][3]) // 2) + x
        yle = int(eyes[0][1]) + (int(eyes[0][3]) // 2) + y
        xre = int(eyes[1][0]) + (int(eyes[1][3]) // 2) + x
        yre = int(eyes[1][1]) + (int(eyes[1][3]) // 2) + y
        print(xle, yle, xre, yre)

        # Calcul des deux distances
        ac = distance(xre, yle, xle, yle)
        ab = distance(xre, yle, xre, yre)

        # Face center
        fCenter = (min(xle, xre) + (ac / 2), yle)

        # Calcul de l'angle de rotation
        c = math.degrees(math.atan(ab / ac))

        # Calcul de proportionalités
        rapport = ac / eyesGap

        inv = 1
        if xle > xre:
            inv = -inv
        if yle > yre:
            inv = -inv

        # Rotation de l'image
        img_vierge = rotate_image(img_vierge, inv * c)
        img = rotate_image(img, inv * c)
        cv2.imshow('Rotated Image', img)
        cv2.waitKey()

        # On rogne l'image
        img_vierge = crop(img_vierge, fCenter, rapport)
        img = crop(img, fCenter, rapport)

        cv2.imshow('Croped Image', img)
        cv2.waitKey()

        # On redimensionne l'image en 178*218
        img_vierge = resize(img_vierge)
        img = resize(img)
        # Displaying the image
        cv2.imshow('Detected Face Image', img_vierge)
        # Waiting for escape key for image to close
        cv2.waitKey()
    else:
        print('AUCUN OEIL DETECTE')

    # Creating Rectangle around face
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imwrite(base + '/Modified/' + img_name + '.jpg', img_vierge)

# Displaying the image
# cv2.imshow('Detected Face Image', img)

# Waiting for escape key for image to close
# cv2.waitKey()
