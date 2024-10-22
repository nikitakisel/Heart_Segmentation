import sys

import heart
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QWidget, QLineEdit, QPushButton, QLabel, \
    QApplication, QFileDialog


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        # You must call the super class method
        super().__init__(parent)

        self.file_name = None
        self.setFixedSize(QSize(1436, 780))  # Set sizes
        self.setWindowTitle("Pathology Warden")  # Set the window title
        central_widget = QWidget(self)  # Create a central widget
        self.setCentralWidget(central_widget)  # Install the central widget

        grid_layout = QGridLayout(self)  # Create QGridLayout
        central_widget.setLayout(grid_layout)  # Set this layout in central widget

        self.img_label = QLabel(self)
        self.img_label.resize(1136, 780)

        self.num_of_areas_textbox = QLineEdit(self)
        self.num_of_areas_textbox.setPlaceholderText('Кол-во областей')
        self.num_of_areas_textbox.move(1170, 120)
        self.num_of_areas_textbox.resize(220, 30)

        self.max_points_count_in_one_area_textbox = QLineEdit(self)
        self.max_points_count_in_one_area_textbox.setPlaceholderText('MAX точек в области')
        self.max_points_count_in_one_area_textbox.move(1170, 160)
        self.max_points_count_in_one_area_textbox.resize(220, 30)

        self.import_button = QPushButton('Выбрать файл', self)
        self.import_button.move(1165, 220)
        self.import_button.resize(230, 30)
        self.import_button.clicked.connect(self.load_image)

        self.explore_button = QPushButton('Исследовать', self)
        self.explore_button.move(1165, 270)
        self.explore_button.resize(230, 30)
        self.explore_button.clicked.connect(self.explore_image)

    def load_image(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', 'img/', "Image files (*.png *.jpg)")
        img = QPixmap(self.file_name)
        self.img_label.setPixmap(img)

    def explore_image(self):
        heart.main(int(self.num_of_areas_textbox.text()), int(self.max_points_count_in_one_area_textbox.text()))


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
