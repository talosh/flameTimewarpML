import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtGui import QTextCursor, QFont
import time
from tqdm import tqdm
import io

# Custom stream object to capture output
class Stream(QObject):
    newText = Signal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass

# A thread that does some work and produces output
class Worker(QThread):
    def run(self):
        print ('hello')
        for i in tqdm(range(100),
                      file=sys.stdout,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', 
                      # ascii=f' {chr(0x2588)}',
                      ascii=False,
                      ncols=50):
            time.sleep(0.1)  # Simulate work

# Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Redirect sys.stdout and sys.stderr
        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)

        self.worker = Worker()
        self.worker.finished.connect(self.onWorkerFinished)
        self.worker.start()

        self.last_progress_line = None  # Keep track of the last progress line

    def initUI(self):
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        font = QFont("Monaco", 12)  # Generic monospace font
        self.text_edit.setFont(font)
        self.text_edit.setReadOnly(True)
        
        layout.addWidget(self.text_edit)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Capture tqdm Output')
        self.show()

    def onUpdateText(self, text):
        text = text.rstrip('\n')
        # Check for carriage return indicating a progress update
        if '\r' in text:
            text = text.rstrip('\r')
            if self.last_progress_line is not None:
                # Remove the last progress update
                self.text_edit.moveCursor(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                self.text_edit.textCursor().removeSelectedText()
                self.text_edit.moveCursor(QTextCursor.End)
                self.text_edit.textCursor().deletePreviousChar()  # Remove newline left after text removal
            self.last_progress_line = text
        else:
            pass
            # Not a progress line, so append normally and reset progress tracking
            # self.last_progress_line = None
            # text = text + '\n'  # Add newline for regular prints

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)  # Insert the text at the end
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def onWorkerFinished(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
