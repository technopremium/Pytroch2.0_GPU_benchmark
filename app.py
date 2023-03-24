import sys
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit
from PyQt5.QtCore import QThread, pyqtSignal
import trainer

class TrainerThread(QThread):
    def run(self):
        trainer.setup()
        self.trainer_instance = trainer.setup_trainer()
        self.trainer_instance.train()

    def stop(self):
        self.trainer_instance.stop()

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

        self.stop_benchmark_button = QPushButton("Stop")
        self.stop_benchmark_button.clicked.connect(self.stop_benchmark)

        layout.addWidget(self.console)
        layout.addWidget(self.start_benchmark_button)
        layout.addWidget(self.stop_benchmark_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Redirect stdout to the console
        sys.stdout = self

    def start_benchmark(self):
        self.trainer_thread = TrainerThread()
        self.trainer_thread.start()

    def stop_benchmark(self):
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.trainer_thread.trainer_instance.stop()


    def write(self, text):
        self.console.appendPlainText(text)

    def flush(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = App()
    main_win.show()
    sys.exit(app.exec())
