import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal
import time
from tqdm import tqdm
import io
import contextlib
import signal

# A thread that does some work and emits output
class Worker(QThread):
    output = Signal(str)
    
    def run(self):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            for i in tqdm(range(100)):
                print(f"Step {i}")  # Example print statement
                time.sleep(0.1)  # Simulate work
                self.output.emit(f.getvalue())
                f.truncate(0)
                f.seek(0)

# Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.worker = Worker()
        self.worker.output.connect(self.display_output)
        self.worker.start()

    def initUI(self):
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        
        layout.addWidget(self.text_edit)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Terminal-like Application')
        self.show()

    def display_output(self, text):
        self.text_edit.moveCursor(self.text_edit.textCursor().End)
        self.text_edit.insertPlainText(text)
        QApplication.processEvents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    window = MainWindow()
    sys.exit(app.exec())
