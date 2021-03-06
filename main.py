
import time
import datetime
from threading import Thread

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib.pyplot as plt

import h5py
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from yolo3.model import *
from yolo3.detect import *

from utils.image import *
from utils.datagen import *


class Ui_Form(object):
    veri=1
    def __init__(self):
        self.capture = None
        self.FPS_MS = None
        self.change_pixmap_signal = pyqtSignal(np.ndarray)

    def setupUi(self, Form):

        Form.setObjectName("Form")
        Form.resize(1918, 1080)
        self.fileref = QtWidgets.QTextEdit()
        self.fileref.setGeometry(QtCore.QRect(0, 0, 0, 0))
        self.fileref.setObjectName("fileref")


        self.dataref = QtWidgets.QTextEdit()
        self.dataref.setGeometry(QtCore.QRect(0, 0, 0, 0))
        self.dataref.setObjectName("dataref")
        self.camera_Button = QtWidgets.QPushButton(Form)
        self.camera_Button.setGeometry(QtCore.QRect(60, 510, 93, 28))
        self.camera_Button.setObjectName("camera_Button")
        self.one_Method = QtWidgets.QPushButton(Form)
        self.one_Method.setGeometry(QtCore.QRect(180, 510, 93, 28))
        self.one_Method.setObjectName("pushButton_2")
        self.dosya_Sec = QtWidgets.QPushButton(Form)
        self.dosya_Sec.setGeometry(QtCore.QRect(540, 510, 93, 28))
        self.dosya_Sec.setObjectName("dosya_Sec")
        self.detec_screen = QtWidgets.QListWidget(Form)
        self.detec_screen.setGeometry(QtCore.QRect(20, 10, 641, 481))
        self.detec_screen.setObjectName("detec_screen")
        item = QtWidgets.QListWidgetItem()
        self.detec_screen.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.detec_screen.addItem(item)
        self.detec_screen.setAutoScroll(True)

        self.dosya_Sec.clicked.connect(self.dosya_sec)
        self.camera_Button.clicked.connect(self.baslat)
        self.one_Method.clicked.connect(self.first_method)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.camera_Button.setText(_translate("Form", "Kamera"))
        self.one_Method.setText(_translate("Form", "Video"))
        self.dosya_Sec.setText(_translate("Form", "Dosya Se??"))

    def prepare_model(approach):
        global input_shape, class_names, anchor_boxes, num_classes, num_anchors, model
        print(approach)

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

    def baslat(self):


        vid = cv2.VideoCapture(0)

        while (True):
            an = datetime.datetime.now()
            tarih = datetime.datetime.ctime(an)
            ret, frame = vid.read()

            # img = frame

            img = letterbox_image(frame, input_shape)

            img = Ui_Form.get_detection(frame)


            cv2.imshow('frame', img)
            fihrist = open("data.txt", "r")
            refdata=fihrist.readline()
            if(refdata[0]=="W"):
                refdata="Ki??i Alg??land?? Ba??ar?? Oran?? : "+str(tarih)+"---->"+refdata
            elif(refdata[0]=="H"):
                refdata="Kask Alg??land??, Ba??ar?? Oran?? : "+str(tarih)+"---->"+refdata
            elif(refdata[0]=="V"):
                refdata="Yelek Alg??land??, Ba??ar?? Oran?? : "+str(tarih)+"---->"+refdata
            self.detec_screen.addItem(refdata)
            self.detec_screen.addItem(">>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<")
            self.detec_screen.scrollToBottom()
            dosya = open("data.txt", "w")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.detec_screen.addItem("KAMERA KAPATILDI!!!")
                break

        vid.release()
        cv2.destroyAllWindows()

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
            image_shape=(ih, iw),
            input_shape=(416, 416),
            max_boxes=10,
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

    def read_model(approach,self):
        filename = f'model-data\weights\pictor-ppe-v302-a{approach}-yolo-v3-weights.h5'

        with h5py.File(filename, "r") as f:
            # List all groupsq

            print()
            print("Keys: %s" % f.keys())


            a_group_key = list(f.keys())[0]

            # Get the data
            data = list(f[a_group_key])
            return "Keys: %s" % f.keys()

    #read_model(approach=3)

    def dosya_sec(self):
        filename = QFileDialog.getOpenFileName()
        self.fileref.setText(filename[0])
        dosya = open("reference.txt", "w")
        dosya.write(self.fileref.toPlainText())
        dosya.close()
        print(self.fileref.toPlainText())

    def show_frame(self):
        self.frame = letterbox_image(self.frame, input_shape)
        self.frame = self.get_detection(self.frame)
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)

    def first_method(self):
        src = self.fileref.toPlainText()
        vid = cv2.VideoCapture(src)

        while (True):
            an = datetime.datetime.now()
            tarih = datetime.datetime.ctime(an)
            ret, frame = vid.read()

            # img = frame

            img = letterbox_image(frame, input_shape)

            img = Ui_Form.get_detection(frame)

            cv2.imshow('frame', img)
            fihrist = open("data.txt", "r")
            refdata = fihrist.readline()
            if (refdata[0] == "W"):
                refdata = "Ki??i Alg??land?? Ba??ar?? Oran?? : " + str(tarih) + "---->" + refdata
            elif (refdata[0] == "H"):
                refdata = "Kask Alg??land??, Ba??ar?? Oran?? : " + str(tarih) + "---->" + refdata
            elif (refdata[0] == "V"):
                refdata = "Yelek Alg??land??, Ba??ar?? Oran?? : " + str(tarih) + "---->" + refdata
            self.detec_screen.addItem(refdata)
            self.detec_screen.addItem(">>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<")
            self.detec_screen.scrollToBottom()
            dosya = open("data.txt", "w")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.detec_screen.addItem("V??DEO KAPATILDI!!!")
                break

        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
