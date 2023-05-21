import evaluateAlgorithm
from machineLearning import *
from processImage import *
from skimage import morphology

# def min_max_snap(src_img, limit=7.):
#     w, h = src_img.shape
#     new_img = np.zeros_like(src_img)
#     for i in range(w):
#         for j in range(h):
#             # img[i][j] = int(img[i][j] * 255)
#             new_img[i][j] = 255 if src_img[i][j] > limit else 0
#
#     return new_img
#
# def process_image(img, scale=1.0):
#     processed_img = pi.resize_img(img.copy(), scale)
#     _, green_channel, _ = cv2.split(processed_img)  # split img into 3 channels: blue, green, red
#     cv2.imwrite('green_channel.jpg', green_channel)
#
#     mask = pi.prepare_mask(green_channel, 25)  # prepare mask to remove outer circle from image
#     cv2.imwrite('mask.jpg', mask)
#
#     # sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     #sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#     #processed_img = cv2.filter2D(green_channel, -1, sharpen_filter)  # sharpen image
#
#     processed_img = cv2.medianBlur(green_channel, 7)
#     cv2.imwrite('blured_img.jpg', processed_img)
#
#     processed_img = (frangi(processed_img, gamma=0.7, beta=0.15))
#     processed_img = (processed_img * 255).astype(np.uint8)
#     processed_img = cv2.medianBlur(processed_img, 3)
#     processed_img = min_max_snap(processed_img, 35)
#     cv2.imwrite('frangi.jpg', processed_img)
#
#     kernel = np.ones((6, 6), np.uint8)
#     processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)
#     kernel = np.ones((3, 3), np.uint8)
#     processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)
#     cv2.imwrite('post-open.jpg', processed_img)
#
#     processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)
#     processed_img = processed_img.astype(np.uint8)
#     cv2.imwrite('final-img.jpg', processed_img)
#     return processed_img

# a = np.array([[0, 255, 0, 0, 0, 0],
#               [255, 255, 255, 0, 255, 0],
#               [255, 255, 255, 0, 0, 0]], int)
# a = np.array(a, bool)
# b = morphology.remove_small_objects(a, min_size=3)
# b = b * 255
# print(a)
# print(b)

train_imgs = []
expert_imgs = []
for i in range(4, 5):
    train_img_path = 'data/Image_0' + str(i) + 'L.jpg'
    train_img = cv2.imread(train_img_path)
    train_img = ProcessImage.preprocess(train_img)
    train_imgs.append(train_img)

    expert_img_path = 'goal/Image_0' + str(i) + 'L_1stHO.png'
    expert_img = cv2.imread(expert_img_path, cv2.IMREAD_GRAYSCALE)

    # h, w = expert_img.shape
    # scale = 0.25
    # h = int(h * scale)
    # w = int(w * scale)
    # expert_img = cv2.resize(expert_img, (w, h), interpolation=cv2.INTER_LINEAR)

    expert_imgs.append(expert_img)

predict_imgs = []
expert_predict_imgs = []
for i in range(5, 6):
    predict_img_path = 'data/Image_0' + str(i) + 'L.jpg'
    predict_img = cv2.imread(predict_img_path)
    predict_img = ProcessImage.preprocess(predict_img)
    predict_imgs.append(predict_img)

    expert_img_path = 'goal/Image_0' + str(i) + 'L_1stHO.png'
    expert_predict_img = cv2.imread(expert_img_path, cv2.IMREAD_GRAYSCALE)
    expert_predict_imgs.append(expert_predict_img)


classifier = Classifier(trainImages_x=train_imgs, trainImages_y=expert_imgs)
y_pred = classifier.predict(predict_imgs[0])

# y_pred = cv2.imread('temp.jpg', cv2.IMREAD_GRAYSCALE)
# ret, y_pred = cv2.threshold(y_pred, 127, 255, cv2.THRESH_BINARY)
y_pred = np.array(y_pred, bool)
y_pred = morphology.remove_small_objects(y_pred, min_size=32)
y_pred = y_pred * 255


confusion_img = evaluateAlgorithm.evaluate(y_pred, expert_predict_imgs[0], imbalanced_data=True)
model_img_rgb = cv2.cvtColor(expert_predict_imgs[0],
                             cv2.COLOR_GRAY2BGR)  # while concatenating all arrays have to have same dimension

prediction_img_rgb = cv2.cvtColor(y_pred.astype(np.uint8), cv2.COLOR_GRAY2BGR)

images = np.concatenate((prediction_img_rgb, model_img_rgb, confusion_img), axis=1)
cv2.namedWindow('images', cv2.WINDOW_NORMAL)
cv2.resizeWindow('images', 1800, 600)
cv2.imshow('images', images)

cv2.waitKey(0)
cv2.destroyAllWindows()
