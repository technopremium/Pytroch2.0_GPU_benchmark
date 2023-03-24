import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QPlainTextEdit, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, QDialogButtonBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from trainer import setup, setup_trainer

class TrainerThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            self.log_signal.emit("Setting up...\n")
            setup()

            self.log_signal.emit("Setting up trainer...\n")
            trainer_instance = setup_trainer(self.settings)

            self.log_signal.emit("Starting training...\n")
            trainer_instance.train()
        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}\n")

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Pytorch Benchmark")

        layout = QVBoxLayout()

        self.start_button = QPushButton("Start Benchmarking")
        self.start_button.clicked.connect(self.start_benchmark)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_benchmark)
        layout.addWidget(self.stop_button)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        layout.addWidget(self.settings_button)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.trainer_thread = None

    def start_benchmark(self):
        self.start_button.setEnabled(False)
        self.settings_button.setEnabled(False)

        if self.trainer_thread is not None:
            self.trainer_thread.terminate()
            self.trainer_thread.wait()

        self.trainer_thread = TrainerThread(self.settings)
        self.trainer_thread.log_signal.connect(self.update_console)
        self.trainer_thread.finished.connect(self.on_thread_finished)
        self.trainer_thread.start()

    def stop_benchmark(self):
        if self.trainer_thread is not None:
            self.trainer_thread.terminate()
            self.trainer_thread.wait()

    def update_console(self, text):
        self.console.appendPlainText(text)

    def on_thread_finished(self):
        self.start_button.setEnabled(True)
        self.settings_button.setEnabled(True)

    def open_settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")

        layout = QFormLayout()

        batch_size_spinbox = QSpinBox()
        batch_size_spinbox.setValue(self.settings["per_device_train_batch_size"])
        layout.addRow("Train Batch Size:", batch_size_spinbox)

        eval_batch_size_spinbox = QSpinBox()
        eval_batch_size_spinbox.setValue(self.settings["per_device_eval_batch_size"])
        layout.addRow("Eval Batch Size:", eval_batch_size_spinbox)

        learning_rate_spinbox = QDoubleSpinBox()
        learning_rate_spinbox.setDecimals(8)
        learning_rate_spinbox.setSingleStep(0.00001)
        learning_rate_spinbox.setValue(self.settings["learning_rate"])
        layout.addRow("Learning Rate:", learning_rate_spinbox)

        epochs_spinbox = QSpinBox()
        epochs_spinbox.setValue(self.settings["num_train_epochs"])
        layout.addRow("Epochs:", epochs_spinbox)

        bf16_checkbox = QCheckBox()
        bf16_checkbox.setChecked(self.settings["bf16"])
        layout.addRow("Use BF16:", bf16_checkbox)

        torch_compile_checkbox = QCheckBox()
        torch_compile_checkbox.setChecked(self.settings["torch_compile"])
        layout.addRow("Use Torch Compile:", torch_compile_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        result = dialog.exec_()

        if result == QDialog.Accepted:
            self.settings["per_device_train_batch_size"] = batch_size_spinbox.value()
            self.settings["per_device_eval_batch_size"] = eval_batch_size_spinbox.value()
            self.settings["learning_rate"] = learning_rate_spinbox.value()
            self.settings["num_train_epochs"] = epochs_spinbox.value()
            self.settings["bf16"] = bf16_checkbox.isChecked()
            self.settings["torch_compile"] = torch_compile_checkbox.isChecked()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = App()
    ex.settings = {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "bf16": True,
        "torch_compile": True,
    }
    ex.show()
    sys.exit(app.exec_())

