from PyQt5.QtWidgets import QMessageBox


def show_alert(text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.Ok)
    retval = msg.exec_()


def show_critical(text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.Ok)
    retval = msg.exec_()
