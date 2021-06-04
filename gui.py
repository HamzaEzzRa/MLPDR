from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QAbstractButton, QButtonGroup, QCheckBox, QFileDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QWidget
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot

import os
import cv2
import sys

import arabic_reshaper
from bidi.algorithm import get_display

from detection import PlateDetector
from ocr import PlateReader
from utility import enum

from threading import Thread

os.environ['QT_DEVICE_PIXEL_RATIO'] = '0'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'
os.environ['QT_SCALE_FACTOR'] = '1'

OCR_MODES = enum('TRAINED', 'TESSERACT')

class MainWindow(QMainWindow):
    def on_resize(self):
        self.dimension = self.size()
        width, height = self.dimension.width(), self.dimension.height()

        self.car_image.setGeometry(QtCore.QRect(width * 0.275, height * 0.15, width * 0.3, height * 0.5))
        self.load_image.setGeometry(QtCore.QRect(width * 0.03, height * 0.2, width * 0.125, height * 0.05))
        self.line.setGeometry(QtCore.QRect(width * 0.179, height * 0.1, 20, height))
        self.line_2.setGeometry(QtCore.QRect(0, height * 0.06, width, 16))
        self.car_detection.setGeometry(QtCore.QRect(width * 0.625, height * 0.15, width * 0.3, height * 0.5))
        self.cropped_plat.setGeometry(QtCore.QRect(width * 0.25, height * 0.8, width * 0.2, height * 0.075))
        self.plate_characters.setGeometry(QtCore.QRect(width * 0.49, height * 0.8, width * 0.2, height * 0.075))
        self.start_detection.setGeometry(QtCore.QRect(width * 0.03, height * 0.26, width * 0.125, height * 0.05))
        self.load_video.setGeometry(QtCore.QRect(width * 0.03, height * 0.35, width * 0.125, height * 0.05))
        self.stop_video.setGeometry(QtCore.QRect(width * 0.03, height * 0.41, width * 0.125, height * 0.05))
        self.open_camera.setGeometry(QtCore.QRect(width * 0.03, height * 0.5, width * 0.125, height * 0.05))
        self.close_camera.setGeometry(QtCore.QRect(width * 0.03, height * 0.56, width * 0.125, height * 0.05))
        self.exit.setGeometry(QtCore.QRect(width * 0.03, height * 0.74, width * 0.125, height * 0.05))
        self.line_3.setGeometry(QtCore.QRect(width * 0.21, height * 0.69, width * 0.765, 20))
        self.plate_ocr.setGeometry(QtCore.QRect(width * 0.73, height * 0.8, width * 0.2, height * 0.075))
        self.trained_ocr.move(width * 0.29, height * 0.735)
        self.tesseract_ocr.move(width * 0.29 + 250, height * 0.735)

        label_4_width = self.label_4.fontMetrics().boundingRect(self.label_4.text()).width()
        label_4_height = self.label_4.fontMetrics().boundingRect(self.label_4.text()).height()
        self.label_4.setGeometry(QtCore.QRect((width - label_4_width + 1) * 0.5, height * 0.015, label_4_width + 1, label_4_height))

        label_2_width = self.label_4.fontMetrics().boundingRect(self.label_2.text()).width()
        label_2_height = self.label_4.fontMetrics().boundingRect(self.label_2.text()).height()
        self.label_2.setGeometry(QtCore.QRect(width * 0.21, height * 0.1, label_2_width, label_2_height))

        label_3_width = self.label_4.fontMetrics().boundingRect(self.label_3.text()).width()
        label_3_height = self.label_4.fontMetrics().boundingRect(self.label_3.text()).height()
        self.label_3.setGeometry(QtCore.QRect(width * 0.21, height * 0.73, label_3_width, label_3_height))

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        self.on_resize()

    def setup(self, width, height):
        self.setObjectName("window")
        self.resize(width, height)
        self.setMinimumSize(512, 512)
        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setObjectName("central_widget")
        self.car_image = QtWidgets.QLabel(self.central_widget)
        self.car_image.setText("")
        self.car_image.setScaledContents(True)
        self.car_image.setObjectName("car_image")
        self.load_image = QtWidgets.QPushButton(self.central_widget)
        self.load_image.setObjectName("load_image")
        self.line = QtWidgets.QFrame(self.central_widget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.central_widget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.car_detection = QtWidgets.QLabel(self.central_widget)
        self.car_detection.setText("")
        self.car_detection.setScaledContents(True)
        self.car_detection.setObjectName("car_detection")
        self.cropped_plat = QtWidgets.QLabel(self.central_widget)
        self.cropped_plat.setText("")
        self.cropped_plat.setObjectName("cropped_plat")
        self.label_4 = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.plate_characters = QtWidgets.QLabel(self.central_widget)
        self.plate_characters.setText("")
        self.plate_characters.setObjectName("plate_characters")
        self.start_detection = QtWidgets.QPushButton(self.central_widget)
        self.start_detection.setObjectName("start_detection")
        self.load_video = QtWidgets.QPushButton(self.central_widget)
        self.load_video.setObjectName("load_video")
        self.stop_video = QtWidgets.QPushButton(self.central_widget)
        self.stop_video.setObjectName("stop_video")
        self.open_camera = QtWidgets.QPushButton(self.central_widget)
        self.open_camera.setObjectName("open_camera")
        self.close_camera = QtWidgets.QPushButton(self.central_widget)
        self.close_camera.setObjectName("close_camera")
        self.exit = QtWidgets.QPushButton(self.central_widget)
        self.exit.setObjectName("exit")
        self.line_3 = QtWidgets.QFrame(self.central_widget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_2 = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.central_widget)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.plate_ocr = QtWidgets.QLabel(self.central_widget)
        self.plate_ocr.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.plate_ocr.setAlignment(QtCore.Qt.AlignCenter)
        self.plate_ocr.setObjectName("plate_ocr")
        font = self.plate_ocr.font()
        font.setPointSize(20)
        self.plate_ocr.setFont(font)
        self.car_image.setStyleSheet("border: 1px solid black;") 
        self.car_detection.setStyleSheet("border: 1px solid black;") 
        self.cropped_plat.setStyleSheet("border: 1px solid black;") 
        self.plate_characters.setStyleSheet("border: 1px solid black;") 
        self.plate_ocr.setStyleSheet("border: 1px solid black;") 

        self.load_image.clicked.connect(self.on_click_load)
        self.exit.clicked.connect(self.exit_app)
        self.open_camera.clicked.connect(self.open_cam)
        self.close_camera.clicked.connect(self.close_cam)
        self.load_video.clicked.connect(self.on_click_load_vd)
        self.start_detection.clicked.connect(self.trained_anpr)
        self.stop_video.clicked.connect(self.stop_video_playing)

        self.trained_ocr = QCheckBox(self.central_widget)
        self.trained_ocr.setObjectName("trained_ocr")
        self.trained_ocr.setText("Moroccan Plate (Custom OCR)")

        self.tesseract_ocr = QCheckBox(self.central_widget)
        self.tesseract_ocr.setObjectName("tesseract_ocr")
        self.tesseract_ocr.setText("General Plate (Tesseract-OCR)")

        self.ocrButtonGroup = QButtonGroup()
        self.ocrButtonGroup.addButton(self.trained_ocr, 1)
        self.ocrButtonGroup.addButton(self.tesseract_ocr, 2)
        self.ocrButtonGroup.buttonClicked[QAbstractButton].connect(self.ocr_switch)

        self.ocr_mode = OCR_MODES.TRAINED
        self.trained_ocr.setChecked(True)

        self.setCentralWidget(self.central_widget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslate()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.image_path = ""

        self.detector = PlateDetector()
        self.detector.load_model("./weights/detection/yolov3-detection_final.weights", "./weights/detection/yolov3-detection.cfg")

        self.reader = PlateReader()
        self.reader.load_model("./weights/ocr/yolov3-ocr_final.weights", "./weights/ocr/yolov3-ocr.cfg")

    def ocr_switch(self, btn):
        if (btn.text() == self.trained_ocr.text()):
            self.ocr_mode = OCR_MODES.TRAINED
        elif (btn.text() == self.tesseract_ocr.text()):
            self.ocr_mode = OCR_MODES.TESSERACT
        
        self.clear_ocr()

    def retranslate(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("window", "MLPDR"))

        self.label_4.setText(_translate("window", "Moroccan License Plate Detection & Recognition"))

        self.label_2.setText(_translate("window", "Detection :"))

        self.label_3.setText(_translate("window", "Recognition :"))

        self.load_image.setText(_translate("window", "Load Image"))
        self.start_detection.setText(_translate("window", "Start Detection"))
        self.load_video.setText(_translate("window", "Load Video"))
        self.stop_video.setText(_translate("window", "Stop Video"))
        self.open_camera.setText(_translate("window", "Open Camera"))
        self.close_camera.setText(_translate("window", "Close Camera"))
        self.exit.setText(_translate("window", "Exit"))

    def popup_close(self, input):
        if input.text() == '&Yes':
            sys.exit(1)

    def exit_app(self):
        message = QMessageBox()
        message.setWindowTitle('Warning')
        message.setText('Are you sure you want to exit ?')
        message.setIcon(QMessageBox.Warning)
        message.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        message.setDefaultButton(QMessageBox.No)
        message.buttonClicked.connect(self.popup_close)
        x = message.exec_()

    def closeEvent(self, event):
        self.exit_app()

    def on_click_load(self):
        self.stop_video_playing()
        self.close_cam()
        self.clear_ocr()
        self.image_path = ""
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file (*.jpg *.png)")
        self.image_path = image[0]
        pixmap = QPixmap(self.image_path)
        self.car_image.setScaledContents(True)
        t = QtGui.QTransform()
        rotated_pixmap = pixmap.transformed(t)
        self.car_image.setPixmap(rotated_pixmap)

    def on_click_load_vd(self):
        self.close_cam()
        self.clear_ocr()
        self.play = True
        vd = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Video file (*.mp4)")
        videoPath = vd[0]
        videoCapture = cv2.VideoCapture(videoPath)
        while (videoCapture.isOpened() and self.play):
            ret, frame = videoCapture.read()
            if ret:
                cv2.imwrite('./tmp/frame.jpg', frame)
                self.car_image.setPixmap(QtGui.QPixmap('./tmp/frame.jpg'))
                self.car_image.setScaledContents(True)

                image, height, width, channels = self.detector.load_image('./tmp/frame.jpg')
                blob, outputs = self.detector.detect_plates(image)
                boxes, confidences, class_ids = self.detector.get_boxes(outputs, width, height, threshold=0.3)
                plate_img, LpImg = self.detector.draw_labels(boxes, confidences, class_ids, image)
                if len(LpImg):
                    cv2.imwrite('./tmp/car_box.jpg', plate_img)
                    cv2.imwrite('./tmp/plate_box.jpg', LpImg[0])
                    self.car_detection.setPixmap(QtGui.QPixmap('./tmp/car_box.jpg'))
                    self.car_detection.setScaledContents(True)
                    self.cropped_plat.setPixmap(QtGui.QPixmap('./tmp/plate_box.jpg'))
                    self.cropped_plat.setScaledContents(True)

                    self.apply_ocr()
                else:
                    self.car_detection.setPixmap(QtGui.QPixmap('./tmp/frame.jpg'))
                    self.car_detection.setScaledContents(True)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        videoCapture.release()
        cv2.destroyAllWindows()

    def stop_video_playing(self):
        self.play = False

    def open_cam(self):
        self.stop_video_playing()
        self.clear_ocr()
        capture = cv2.VideoCapture(0)
        self.recorded = True
        while(self.recorded):
            ret, frame = capture.read()

            cv2.imwrite('./tmp/frame.jpg', frame)
            self.car_image.setPixmap(QtGui.QPixmap('./tmp/frame.jpg'))
            self.car_image.setScaledContents(True)

            image, height, width, channels = self.detector.load_image('./tmp/frame.jpg')
            blob, outputs = self.detector.detect_plates(image)
            boxes, confidences, class_ids = self.detector.get_boxes(outputs, width, height, threshold=0.3)
            plate_img, LpImg = self.detector.draw_labels(boxes, confidences, class_ids, image)
            if len(LpImg):
                cv2.imwrite('./tmp/car_box.jpg', plate_img)
                cv2.imwrite('./tmp/plate_box.jpg', LpImg[0])
                self.car_detection.setPixmap(QtGui.QPixmap('./tmp/car_box.jpg'))
                self.car_detection.setScaledContents(True)
                self.cropped_plat.setPixmap(QtGui.QPixmap('./tmp/plate_box.jpg'))
                self.cropped_plat.setScaledContents(True)

                self.apply_ocr()
            else:
                self.car_detection.setPixmap(QtGui.QPixmap('./tmp/frame.jpg'))
                self.car_detection.setScaledContents(True)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def close_cam(self):
        self.recorded = False

    def apply_ocr(self):
        if (self.ocr_mode == OCR_MODES.TRAINED):
            image, height, width, channels = self.reader.load_image('./tmp/plate_box.jpg')
            blob, outputs = self.reader.read_plate(image)
            boxes, confidences, class_ids = self.reader.get_boxes(outputs, width, height, threshold=0.3)
            segmented, plate_text = self.reader.draw_labels(boxes, confidences, class_ids, image)
            cv2.imwrite('./tmp/plate_segmented.jpg', segmented)
            if (len(plate_text) > 0):
                self.plate_characters.setPixmap(QtGui.QPixmap('./tmp/plate_segmented.jpg'))
                self.plate_characters.setScaledContents(True)
                self.plate_ocr.setText(get_display(arabic_reshaper.reshape(plate_text)))
        elif (self.ocr_mode == OCR_MODES.TESSERACT):
            plate_text = self.reader.tesseract_ocr('./tmp/plate_box.jpg')
            if (len(plate_text) > 0):
                self.plate_characters.clear()
                self.plate_ocr.setText(plate_text)

    def clear_ocr(self):
        self.plate_ocr.clear()
        self.plate_characters.clear()

    def trained_anpr(self):
        if (self.image_path == ""):
            return
        
        image, height, width, channels = self.detector.load_image(self.image_path)
        blob, outputs = self.detector.detect_plates(image)
        boxes, confidences, class_ids = self.detector.get_boxes(outputs, width, height, threshold=0.3)
        plate_img, LpImg = self.detector.draw_labels(boxes, confidences, class_ids, image)
        if len(LpImg):
            cv2.imwrite('./tmp/car_box.jpg', plate_img)
            cv2.imwrite('./tmp/plate_box.jpg', LpImg[0])
            self.car_detection.setPixmap(QtGui.QPixmap('./tmp/car_box.jpg'))
            self.car_detection.setScaledContents(True)
            self.cropped_plat.setPixmap(QtGui.QPixmap('./tmp/plate_box.jpg'))
            self.cropped_plat.setScaledContents(True)
        
            self.apply_ocr()
