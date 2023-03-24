import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit
from PyQt5.QtCore import QThread, pyqtSignal
import trainer  # Assuming the provided code is saved as trainer.py


class TrainerThread(QThread):
    output = pyqtSignal(str)

    def run(self):
        trainer.setup()
        trainer_instance = trainer.setup_trainer()
        trainer_instance.train()


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyTorch GPU Benchmark GUI")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)

        self.start_benchmark_button = QPushButton("Start Benchmarking")
        self.start_benchmark_button.clicked.connect(self.start_benchmark)

        layout.addWidget(self.console)
        layout.addWidget(self.start_benchmark_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_benchmark(self):
        self.trainer_thread = TrainerThread()
        self.trainer_thread.start()

    def append_output(self, text):
        self.console.appendPlainText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = App()
    main_win.show()
    sys.exit(app.exec())
