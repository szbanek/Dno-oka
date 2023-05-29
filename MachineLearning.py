import math
import cv2
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import local_binary_pattern
from processImage import ProcessImage
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import mahotas


class Classifier:
    def __init__(self, trainImages_x, trainImages_y, n_neighbors=11, sliceSize=5):
        self.sliceSize = sliceSize
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        self.x_train, self.y_train = [], []
        print('Iterating through train images')

        # slice obrazu oraz obliczanie hu momentów dla każdej jego części
        for img in trainImages_x:
            img_sliced = ProcessImage.slice_img(img, self.sliceSize, all=False)
            for img_part in img_sliced:
                x = np.array([])

                huMoments = ProcessImage.get_hu(img_part)
                x = np.append(x, huMoments)

                # lbp = local_binary_pattern(img_part, 8*5, 5, method='ror')
                # lbp = sum(lbp)
                # x = np.append(x, lbp)

                # var = math.sqrt(np.var(img_part))
                # x = np.append(x, var)

                zm = mahotas.features.zernike_moments(img_part, sliceSize)
                zm = np.array([zm[2], zm[4], zm[6], zm[12], zm[20]])
                x = np.append(x, zm)

                self.x_train.append(x)
                if len(self.x_train) % 1000 == 0:
                    print(len(self.x_train))

        print('Iterating through train expert images')
        # slice obrazu eksperckiego i wybieranie jego centralnego punktu
        for img in trainImages_y:
            img_sliced = ProcessImage.slice_img(img, self.sliceSize, all=False)
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
            x = np.array([])

            huMoments = ProcessImage.get_hu(img_part)
            x = np.append(x, huMoments)

            # lbp = local_binary_pattern(img_part, 8*5, 5, method='ror')
            # lbp = sum(lbp)
            # x = np.append(x, lbp)

            # var = math.sqrt(np.var(img_part))
            # x = np.append(x, var)

            zm = mahotas.features.zernike_moments(img_part, self.sliceSize)
            zm = np.array([zm[2], zm[4], zm[6], zm[12], zm[20]])
            x = np.append(x, zm)

            testCases.append(x)

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