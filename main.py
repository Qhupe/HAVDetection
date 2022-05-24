import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

from keras.layers import Input

from yolo3.model import *
from yolo3.detect import *

from utils.image import *
from utils.datagen import *
from utils.fixes import *

def prepare_model(approach):
    global input_shape, class_names, anchor_boxes, num_classes, num_anchors, model

    input_shape = (416, 416)

    if approach == 1:
        class_names = ['H', 'V', 'W']

    elif approach == 2:
        class_names = ['W', 'WH', 'WV', 'WHV']

    elif approach == 3:
        class_names = ['W']

    else:
        raise NotImplementedError('Approach should be 1, 2, or 3')

    if approach == 1:
        anchor_boxes = np.array(
            [
                np.array([[76, 59], [84, 136], [188, 225]]) / 32,
                np.array([[25, 15], [46, 29], [27, 56]]) / 16,
                np.array([[5, 3], [10, 8], [12, 26]]) / 8
            ],
            dtype='float64'
        )
    else:
        anchor_boxes = np.array(
            [
                np.array([[73, 158], [128, 209], [224, 246]]) / 32,
                np.array([[32, 50], [40, 104], [76, 73]]) / 16,
                np.array([[6, 11], [11, 23], [19, 36]]) / 8
            ],
            dtype='float64'
        )

    num_classes = len(class_names)
    num_anchors = anchor_boxes.shape[0] * anchor_boxes.shape[1]

    input_tensor = Input(shape=(input_shape[0], input_shape[1], 3))
    num_out_filters = (num_anchors // 3) * (5 + num_classes)

    model = yolo_body(input_tensor, num_out_filters)

    weight_path = f'model-data\weights\pictor-ppe-v302-a{approach}-yolo-v3-weights.h5'
    model.load_weights(weight_path)

def get_detection(img):
    act_img = img.copy()

    ih, iw = act_img.shape[:2]

    img = letterbox_image(img, input_shape)
    img = np.expand_dims(img, 0)
    image_data = np.array(img) / 255.

    prediction = model.predict(image_data)

    boxes = detection(
        prediction,
        anchor_boxes,
        num_classes,
        image_shape = (ih, iw),
        input_shape = (416,416),
        max_boxes = 10,
        score_threshold=0.3,
        iou_threshold=0.45,
        classes_can_overlap=False)

    boxes = boxes[0].numpy()

    return draw_detection(act_img, boxes, class_names)

def plt_imshow(img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')

prepare_model(approach=1)

def read_model(approach):
    filename = f'model-data\weights\pictor-ppe-v302-a{approach}-yolo-v3-weights.h5'

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])

#read_model(approach=3)

vid = cv2.VideoCapture(0)

while (True):
    ret, frame = vid.read()


    #img = frame

    img = letterbox_image(frame, input_shape)

    img = get_detection(frame)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()