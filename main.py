import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


def highlightFace(net, frame, coef_threshold=0.6):
    frameOpencvDnn = frame.copy()

    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= coef_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            faceBoxes.append([x1, y1, x2, y2, confidence])

            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 11), int(round(frameHeight / 150)), 0)
            label = f"{confidence * 100:.1f}%"
            cv2.putText(frameOpencvDnn, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 11), 2, cv2.LINE_AA)

    return frameOpencvDnn, faceBoxes


class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition")

        # Загрузим модель
        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.btn_select = QPushButton("Выбрать изображение")
        self.btn_select.clicked.connect(self.select_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_select)

        self.setLayout(layout)

        # Камера
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.showing_image = False  # Флаг, показываем ли статичное изображение

    def update_frame(self):
        if self.showing_image:
            return  # если показываем выбранную картинку — не обновляем с камеры

        ret, frame = self.cap.read()
        if not ret:
            return

        frame, _ = highlightFace(self.faceNet, frame)
        self.display_image(frame)

    def display_image(self, img):
        # Конвертируем BGR(OpenCV) в RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def select_image(self):
        # Остановим камеру и таймер
        self.timer.stop()
        self.cap.release()
        self.showing_image = True

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "",
                                                  "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if filename:
            img = cv2.imread(filename)
            if img is None:
                self.image_label.setText("Не удалось загрузить изображение")
                return

            img, _ = highlightFace(self.faceNet, img)
            self.display_image(img)
        else:
            # Если файл не выбран, попробуем снова включить камеру
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.showing_image = False

    def closeEvent(self, event):
        # При закрытии окна освобождаем камеру
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
