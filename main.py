import sys
import mainwindow
from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    mw = mainwindow.MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
