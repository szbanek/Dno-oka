import cv2
import numpy as np
import math


class ProcessImage:

    def slice_img(src_img, size, all=True):
        slices = []
        if all:
            for x in range(size - 1, src_img.shape[0] - size, 1):
                for y in range(size - 1, src_img.shape[1] - size, 1):
                    slices.append(src_img[x:x+size, y:y+size])
        else:
            for x in range(size - 1, src_img.shape[0] - size, 2):
                for y in range(size - 1, src_img.shape[1] - size, 2):
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

        mask = ProcessImage.prepare_mask(img, border_size=5)

        # blurowanie które pozostawia krawedzie ostre, przynajmniej w zalożeniach bo nie widzialem zmiany
        img = cv2.bilateralFilter(img, 9, 11, 11)

        # zwiększenie kontrastu
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(9, 9))
        img = clahe.apply(img)

        return mask, img
