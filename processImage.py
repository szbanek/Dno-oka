import cv2
import numpy as np
import math
from skimage.filters import frangi


class ProcessImage:

    # def __init__(self):
    #     self.mask = None

    def slice_img(src_img, size):
        slices = []
        for x in range(size - 1, src_img.shape[0] - size, 1):
            for y in range(size - 1, src_img.shape[1] - size, 1):
                slices.append(src_img[x:x+size, y:y+size])

        return slices

    def get_hu(src_img_part):
        moments = cv2.moments(src_img_part)
        huMoments = cv2.HuMoments(moments)
        # Log scale hu moments
        for i in range(0, 7):
            if huMoments[i] != 0:
                huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

        flat_huMoments = [item for sublist in huMoments for item in sublist]
        return flat_huMoments


    def extract_green(src_img):
        _, green_channel, _ = cv2.split(src_img)  # split img into 3 channels: blue, green, red
        return green_channel

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

    def prepare_mask(src_img, border_size=3):
        mask = ProcessImage.min_max_snap(src_img, limit=1)
        kernel = np.ones((border_size, border_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def preprocess(src_img):
        img = ProcessImage.resize_img(src_img.copy(), scale=1.0)
        img = ProcessImage.extract_green(img)
        # to ma w zalozeniu pozbyc sie tego swiatla ktore dzieli duze vessele na 2
        # kernel = np.ones((5, 5), np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # zwiększenie kontrastu
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(9, 9))
        img = clahe.apply(img)

        # maska na zewnętrzny okrąg
        # self.mask = self.prepare_mask(img, border_size=25)

        # zastosowanie filtru frangi
        # img = (frangi(img))
        # img = (img * 255).astype(np.uint8)

        # Mialoby pomoc wypelnic vessele po filtru frangi ale nie dziala dobrze xD
        # kernel = np.ones((6, 6), np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # kernel = np.ones((3, 3), np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # binaryzacja obrazu
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        #cv2.imshow('img', img_sliced[26])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return img
