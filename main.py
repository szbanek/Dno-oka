import cv2
import numpy as np
from skimage.filters import frangi, sato
import evaluateAlgorithm
import random as rd


def resize_img(src_img, scale=1.0):
    if scale != 1.0:
        h, w, _ = src_img.shape
        h = int(h * scale)
        w = int(w * scale)
        resized_img = cv2.resize(src_img, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized_img
    else:
        return src_img


def min_max_snap(src_img, limit=7.):
    w, h = src_img.shape
    new_img = np.zeros_like(src_img)
    for i in range(w):
        for j in range(h):
            # img[i][j] = int(img[i][j] * 255)
            new_img[i][j] = 255 if src_img[i][j] > limit else 0

    return new_img


def remove_bright_pixels(src_img, rm_from=120):
    w, h = src_img.shape
    new_img = np.zeros_like(src_img)
    fill_val = sum(src_img[300]) / len(src_img[300])
    for i in range(w):
        for j in range(h):
            new_img[i][j] = 75 if src_img[i][j] > rm_from else src_img[i][j]
    return new_img


def prepare_mask(src_img, size=3):
    mask = min_max_snap(src_img, limit=1)
    kernel = np.ones((size, size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def process_image(img, scale=1.0):
    processed_img = resize_img(img.copy(), scale)
    _, green_channel, _ = cv2.split(processed_img)  # split img into 3 channels: blue, green, red
    cv2.imwrite('green_channel.jpg', green_channel)

    mask = prepare_mask(green_channel, 25)  # prepare mask to remove outer circle from image
    cv2.imwrite('mask.jpg', mask)

    # sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #processed_img = cv2.filter2D(green_channel, -1, sharpen_filter)  # sharpen image

    processed_img = cv2.medianBlur(green_channel, 7)
    cv2.imwrite('blured_img.jpg', processed_img)

    processed_img = (frangi(processed_img, gamma=0.7, beta=0.15))
    processed_img = (processed_img * 255).astype(np.uint8)
    processed_img = cv2.medianBlur(processed_img, 3)
    processed_img = min_max_snap(processed_img, 35)
    cv2.imwrite('frangi.jpg', processed_img)

    kernel = np.ones((6, 6), np.uint8)
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('post-open.jpg', processed_img)

    processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)
    processed_img = processed_img.astype(np.uint8)
    cv2.imwrite('final-img.jpg', processed_img)
    return processed_img


# kernel = np.ones((2, 2), np.uint8)
# IMG = cv2.dilate(green_img, kernel, iterations=1)
# th3 = cv2.GaussianBlur(th3, (3, 3), 0)
# ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# th3 = np.asarray(th3, dtype=np.uint8)  # inaczej wyrzuca error przy adaptivethreshold jakby th3 nie było w int8 nw
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# IMG = cv2.dilate(green_img, kernel, iterations=1)
# th3 = cv2.erode(th3, kernel, iterations=1)
# th3 = frangi(th3)

model_img = cv2.imread('goal/Image_01L_1stHO.png', cv2.IMREAD_GRAYSCALE)
original_img = cv2.imread('data/Image_01L.jpg')
prediction_img = process_image(original_img, scale=1.0)

confusion_img = evaluateAlgorithm.evaluate(prediction_img, model_img, imbalanced_data=True)
model_img_rgb = cv2.cvtColor(model_img,
                             cv2.COLOR_GRAY2BGR)  # while concatenating all arrays have to have same dimension
prediction_img_rgb = cv2.cvtColor(prediction_img, cv2.COLOR_GRAY2BGR)

images = np.concatenate((prediction_img_rgb, model_img_rgb, confusion_img), axis=1)
cv2.namedWindow('images', cv2.WINDOW_NORMAL)
cv2.resizeWindow('images', 1800, 600)
cv2.imshow('images', images)

cv2.waitKey(0)
cv2.destroyAllWindows()
