import cv2
from sklearn.neighbors import KNeighborsClassifier
from processImage import ProcessImage
import numpy as np
from imblearn.under_sampling import RandomUnderSampler


class Classifier:
    def __init__(self, trainImages_x, trainImages_y, n_neighbors=5, sliceSize=5):
        self.sliceSize = sliceSize
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.x_train, self.y_train = [], []
        print('Iterating through train images')
        # slice obrazu oraz obliczanie hu momentów dla każdej jego części
        for img in trainImages_x:
            img_sliced = ProcessImage.slice_img(img, self.sliceSize)
            for img_part in img_sliced:
                flat_img = [item for sublist in img_part for item in sublist]
                while len(flat_img) < self.sliceSize * self.sliceSize:
                    flat_img.append(flat_img[0])

                moments = list(cv2.moments(img_part).values())
                huMoments = ProcessImage.get_hu(img_part)
                allMoments = np.append(moments, huMoments)

                self.x_train.append(huMoments)

        print('Iterating through train expert images')
        # slice obrazu eksperckiego i wybieranie jego centralnego punktu
        for img in trainImages_y:
            img_sliced = ProcessImage.slice_img(img, self.sliceSize)
            for img_part in img_sliced:
                center_pixel = img_part[self.sliceSize//2][self.sliceSize//2]
                self.y_train.append(center_pixel)

        print('Undersampling...')
        rus = RandomUnderSampler(random_state=0)
        self.x_train, self.y_train = rus.fit_resample(self.x_train, self.y_train)

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        print('Training...')
        self.model.fit(self.x_train, self.y_train)

    def predict(self, predictImage):
        print('Preparing predict image')
        img_sliced = ProcessImage.slice_img(predictImage, self.sliceSize)
        testCases = []
        for img_part in img_sliced:
            flat_img = [item for sublist in img_part for item in sublist]
            while len(flat_img) < self.sliceSize * self.sliceSize:
                flat_img.append(flat_img[0])

            moments = list(cv2.moments(img_part).values())
            huMoments = ProcessImage.get_hu(img_part)
            allMoments = np.append(moments, huMoments)
            testCases.append(huMoments)

        print('Predicting...')
        y_pred = self.model.predict(testCases)
        print('Predicted.')
        predicted_image = np.zeros_like(predictImage)
        i = 0
        for x in range(self.sliceSize - 1, predictImage.shape[0] - self.sliceSize, 1):
            for y in range(self.sliceSize - 1, predictImage.shape[1] - self.sliceSize, 1):
                predicted_image[x, y] = y_pred[i]
                i += 1
        cv2.imwrite('temp.jpg', predicted_image)
        return predicted_image


