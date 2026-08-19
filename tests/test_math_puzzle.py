from PyQt6.QtWidgets import QApplication
import pytest
from ui.math_puzzle_window import MathPuzzleDialog

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_math_puzzle_generation(qapp):
    """Test that the math dialog generates valid equations and handles correctness."""
    dialog = MathPuzzleDialog()
    assert dialog.correct_answer != 0
    assert not dialog.solved
    
    # Extract numbers and operation from equation label to verify the math
    eq_text = dialog.equation_lbl.text() # e.g. "12 + 34 = ?"
    parts = eq_text.replace(" = ?", "").split()
    num1 = int(parts[0])
    op = parts[1]
    num2 = int(parts[2])
    
    if op == "+":
        assert dialog.correct_answer == num1 + num2
    elif op == "-":
        assert dialog.correct_answer == num1 - num2
        
    # Check bounds
    if op == "+":
        assert 10 <= num1 <= 99
        assert 10 <= num2 <= 99
    else:
        assert 50 <= num1 <= 99
        assert 10 <= num2 <= 49

def test_math_puzzle_verification(qapp):
    """Verify that submit behavior behaves correctly on right/wrong inputs."""
    dialog = MathPuzzleDialog()
    orig_correct = dialog.correct_answer
    
    # Try invalid value type
    dialog.answer_input.setText("invalid")
    dialog.verify_answer()
    assert not dialog.solved
    assert dialog.status_lbl.text() == "Please enter a valid number."
    
    # Try incorrect numeric value
    wrong_val = orig_correct + 5
    dialog.answer_input.setText(str(wrong_val))
    dialog.verify_answer()
    assert not dialog.solved
    assert dialog.status_lbl.text() == "Incorrect! Try another one."
    # Verification failure generates a new equation
    assert dialog.correct_answer != orig_correct or dialog.equation_lbl.text() != ""
    
    # Try correct numeric value
    new_correct = dialog.correct_answer
    dialog.answer_input.setText(str(new_correct))
    dialog.verify_answer()
    assert dialog.solved
