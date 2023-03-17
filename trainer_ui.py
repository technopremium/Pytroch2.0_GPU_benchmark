import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

class TrainerUI(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the user interface layout
        layout = QVBoxLayout()

        # Add input fields for training parameters
        self.learning_rate_input = self.create_input_field("Learning Rate", layout)
        self.epochs_input = self.create_input_field("Number of Epochs", layout)
        self.batch_size_input = self.create_input_field("Batch Size", layout)

        # Add a button to start the training process
        self.train_button = QPushButton("Start Training")
        layout.addWidget(self.train_button)

        # Connect the button to the on_train_button_clicked method
        self.train_button.clicked.connect(self.on_train_button_clicked)

        # Set the layout on the application's window
        self.setLayout(layout)

    def create_input_field(self, label_text, layout):
        label = QLabel(label_text)
        layout.addWidget(label)
        input_field = QLineEdit()
        layout.addWidget(input_field)
        return input_field

    def on_train_button_clicked(self):
        # Get values from input fields
        learning_rate = float(self.learning_rate_input.text())
        epochs = int(self.epochs_input.text())
        batch_size = int(self.batch_size_input.text())

        # Start the training process using the input parameters
        print(f"Training with learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}")
        # TODO: Call the training function with the input parameters

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainerUI()
    window.show()
    sys.exit(app.exec_())
