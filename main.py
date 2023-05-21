from skimage.filters import frangi
import evaluateAlgorithm
from machineLearning import *
from processImage import *
from skimage import morphology


def naive_process_image(img, scale=1.0):
    processed_img = ProcessImage.resize_img(img.copy(), scale)
    _, green_channel, _ = cv2.split(processed_img)  # split img into 3 channels: blue, green, red

    mask = ProcessImage.prepare_mask(green_channel, 5)  # prepare mask to remove outer circle from image

    processed_img = cv2.medianBlur(green_channel, 7)

    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(7, 7))
    processed_img = clahe.apply(processed_img)
    cv2.imwrite('preprocessed.jpg', processed_img)

    processed_img = (frangi(processed_img, gamma=0.8, beta=0.15))
    processed_img = (processed_img * 255).astype(np.uint8)
    cv2.imwrite('processed.jpg', processed_img)

    processed_img = cv2.medianBlur(processed_img, 3)
    processed_img = ProcessImage.min_max_snap(processed_img, 35)

    processed_img = np.array(processed_img, bool)
    processed_img = morphology.remove_small_objects(processed_img, min_size=128)
    processed_img = processed_img * 255
    processed_img = processed_img.astype('uint8')

    kernel = np.ones((5, 5), np.uint8)
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)

    processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)
    processed_img = processed_img.astype(np.uint8)
    cv2.imwrite('postprocessed.jpg', processed_img)
    return processed_img


def naive_vessel_detection(image_no=5):
    input_img_path = 'data/Image_{:02d}L.jpg'.format(image_no)
    input_img = cv2.imread(input_img_path)
    output_image = naive_process_image(input_img)

    expert_img_path = 'goal/Image_{:02d}L_1stHO.png'.format(image_no)
    expert_img = cv2.imread(expert_img_path, cv2.IMREAD_GRAYSCALE)
    confusion_img = evaluateAlgorithm.evaluate(output_image, expert_img, imbalanced_data=True)

    model_img_rgb = cv2.cvtColor(expert_img,
                                 cv2.COLOR_GRAY2BGR)  # while concatenating all arrays have to have same dimension

    prediction_img_rgb = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    images = np.concatenate((prediction_img_rgb, model_img_rgb, confusion_img), axis=1)
    cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('images', 1800, 600)
    cv2.imshow('images', images)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def classificator_vessel_detection(image_no=5, training_set=None, use_previous=False):
    if training_set is None:
        training_set = [1, 2, 3, 4]

    train_imgs = []
    expert_imgs = []
    for training_no in training_set:
        train_img_path = 'data/Image_{:02d}L.jpg'.format(training_no)
        train_img = cv2.imread(train_img_path)
        _, train_img = ProcessImage.preprocess(train_img)
        train_imgs.append(train_img)

        expert_img_path = 'goal/Image_{:02d}L_1stHO.png'.format(training_no)
        expert_img = cv2.imread(expert_img_path, cv2.IMREAD_GRAYSCALE)

        expert_imgs.append(expert_img)

    predict_img_path = 'data/Image_{:02d}L.jpg'.format(image_no)
    predict_img = cv2.imread(predict_img_path)
    mask, predict_img = ProcessImage.preprocess(predict_img)
    cv2.imwrite('preprocessed.jpg', predict_img)

    expert_img_path = 'goal/Image_{:02d}L_1stHO.png'.format(image_no)
    expert_predict_img = cv2.imread(expert_img_path, cv2.IMREAD_GRAYSCALE)

    if use_previous:
        y_pred = cv2.imread('temp.jpg', cv2.IMREAD_GRAYSCALE)
        ret, y_pred = cv2.threshold(y_pred, 127, 255, cv2.THRESH_BINARY)
    else:
        classifier = Classifier(trainImages_x=train_imgs, trainImages_y=expert_imgs)
        y_pred = classifier.predict(predict_img)

    cv2.imwrite('processed.jpg', y_pred)

    y_pred = np.array(y_pred, bool)
    y_pred = morphology.remove_small_objects(y_pred, min_size=64)
    y_pred = y_pred * 255
    y_pred = y_pred.astype('uint8')
    kernel = np.ones((5, 5), np.uint8)
    y_pred = cv2.morphologyEx(y_pred, cv2.MORPH_CLOSE, kernel)

    # apply mask
    y_pred = cv2.bitwise_and(y_pred, y_pred, mask=mask)
    y_pred = y_pred.astype(np.uint8)
    cv2.imwrite('postprocessed.jpg', y_pred)

    confusion_img = evaluateAlgorithm.evaluate(y_pred, expert_predict_img, imbalanced_data=True)
    model_img_rgb = cv2.cvtColor(expert_predict_img,
                                 cv2.COLOR_GRAY2BGR)  # while concatenating all arrays have to have same dimension

    prediction_img_rgb = cv2.cvtColor(y_pred.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    images = np.concatenate((prediction_img_rgb, model_img_rgb, confusion_img), axis=1)
    cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('images', 1800, 600)
    cv2.imshow('images', images)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# naive_vessel_detection(5)
classificator_vessel_detection(5, use_previous=False)
