"""Math Puzzle Verification Dialog.

Prompts the user with a simple math problem to prove they are awake before dismissing alerts.
"""

import random
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import Qt

class MathPuzzleDialog(QDialog):
    """Modal dialog that requires solving a simple math question to close."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cognitive Wake-up Challenge")
        self.setMinimumWidth(350)
        # Remove close/minimize/maximize buttons so they must solve it
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        
        self.solved = False
        self.correct_answer = 0
        
        self.init_ui()
        self.generate_problem()

    def init_ui(self):
        """Build UI elements with theme styling."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title/Instructions
        title = QLabel("Drowsiness Alert Active!")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #dc2626;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel("To turn off the alarm, prove you are awake by solving the math problem below:")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        # Math equation label
        self.equation_lbl = QLabel("")
        self.equation_lbl.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff; margin: 10px 0;")
        self.equation_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.equation_lbl)

        # Input row
        input_layout = QHBoxLayout()
        self.answer_input = QLineEdit()
        self.answer_input.setPlaceholderText("Enter your answer")
        self.answer_input.setStyleSheet("""
            QLineEdit {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 8px 12px;
                color: #ffffff;
                font-size: 15px;
            }
            QLineEdit:focus {
                border-color: #3b82f6;
            }
        """)
        self.answer_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.answer_input.returnPressed.connect(self.verify_answer)
        input_layout.addWidget(self.answer_input)
        layout.addLayout(input_layout)

        # Status feedback label
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #ef4444; font-weight: bold;")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_lbl)

        # Action button
        self.submit_btn = QPushButton("Submit Answer")
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
        self.submit_btn.clicked.connect(self.verify_answer)
        layout.addWidget(self.submit_btn)

    def generate_problem(self):
        """Generates a new random addition or subtraction problem."""
        operation = random.choice(["+", "-"])
        
        if operation == "+":
            num1 = random.randint(10, 99)
            num2 = random.randint(10, 99)
            self.correct_answer = num1 + num2
            self.equation_lbl.setText(f"{num1} + {num2} = ?")
        else:
            num1 = random.randint(50, 99)
            num2 = random.randint(10, 49)
            self.correct_answer = num1 - num2
            self.equation_lbl.setText(f"{num1} - {num2} = ?")
            
        self.answer_input.clear()
        self.answer_input.setFocus()

    def verify_answer(self):
        """Check user's answer, accept dialog if correct, else retry."""
        try:
            user_ans = int(self.answer_input.text().strip())
        except ValueError:
            self.status_lbl.setText("Please enter a valid number.")
            return

        if user_ans == self.correct_answer:
            self.solved = True
            self.accept()
        else:
            self.status_lbl.setText("Incorrect! Try another one.")
            self.generate_problem()

    def closeEvent(self, event):
        """Only allow close if solved."""
        if self.solved:
            event.accept()
        else:
            event.ignore()
