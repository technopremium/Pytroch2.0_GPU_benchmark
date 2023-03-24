import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit
from PyQt5.QtCore import QThread, pyqtSignal
import trainer_code  # Assuming the provided code is saved as trainer_code.py


class TrainerThread(QThread):
    output = pyqtSignal(str)

    def run(self):
        trainer = trainer_code.setup_trainer()
        trainer.train()


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyTorch Training GUI")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)

        self.install_button = QPushButton("Install")
        self.install_button.clicked.connect(self.install)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.start_training)

        layout.addWidget(self.console)
        layout.addWidget(self.install_button)
        layout.addWidget(self.start_training_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def install(self):
        trainer_code.setup()
        self.console.appendPlainText("Installation complete.\n")

    def start_training(self):
        self.trainer_thread = TrainerThread()
        self.trainer_thread.output.connect(self.console.appendPlainText)
        self.trainer_thread.start()

    def append_output(self, text):
        self.console.appendPlainText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = App()
    main_win.show()
    sys.exit(app.exec())
