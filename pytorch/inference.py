try:
    import os
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
    from PySide6.QtCore import QObject, Signal, QThread, Qt 
    from PySide6.QtGui import QTextCursor, QFont, QFontDatabase, QFontInfo
    import time
    from tqdm import tqdm

except:
    pass

class Timewarp():
    def __init__(self, json_info):
        pass

# Custom stream object to capture output
class Stream(QObject):
    newText = Signal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass

# A thread that does some work and produces output
class Worker(QThread):
    def __init__(self, argv, parent=None):
        super(Worker, self).__init__(parent)
        self.argv = argv

    def run(self):
        print ('hello')
        print (self.argv)

        import torch
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            x = torch.randn(4, device=mps_device)
            print (x)
        else:
            print ("MPS device not found.")

        for i in tqdm(range(100),
                      file=sys.stdout,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', 
                      ascii=f' {chr(0x2588)}',
                      # ascii=False,
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

        self.worker = Worker(sys.argv)
        self.worker.finished.connect(self.onWorkerFinished)
        self.worker.start()

        self.last_progress_line = None  # Keep track of the last progress line

    def loadMonospaceFont(self):
        DejaVuSansMono = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'fonts',
            'DejaVuSansMono.ttf'
        )

        font_id = QFontDatabase.addApplicationFont(DejaVuSansMono)
        if font_id == -1:
            all_fonts = QFontDatabase.families()
            monospaced_fonts = []
            for font_family in all_fonts:
                font = QFont(font_family)
                font_info = QFontInfo(font)
                if font_info.fixedPitch():
                    monospaced_fonts.append(font_family)
            font = QFont(monospaced_fonts[0], 11)  # Generic monospace font
            return font
        else:
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            if font_families:
                font_family = font_families[0]  # Get the first family name
                font = QFont(font_family, 11)  # Set the desired size
                return font
            else:
                return None

    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                color: rgb(188, 188, 188); 
                background-color: #292929;
                border: 1px solid #474747;
            }
        """)
        font = self.loadMonospaceFont()
        if font:
            self.text_edit.setFont(font)

        layout.addWidget(self.text_edit)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Capture tqdm Output')
        self.show()

    def onUpdateText(self, text):
        # text = text.rstrip('\n')
        # Check for carriage return indicating a progress update
        if '\r' in text:
            text.replace('\n', '')
            text.replace('\r', '')
            # text = text.rstrip('\n')
            # text = text.rstrip('\r')
            if self.last_progress_line is not None:
                # Remove the last progress update
                self.text_edit.moveCursor(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                self.text_edit.textCursor().removeSelectedText()
                self.text_edit.moveCursor(QTextCursor.End)
                self.text_edit.textCursor().deletePreviousChar()  # Remove newline left after text removal
            self.last_progress_line = text
        else:
            pass
            # text = text + '\n'
            # Not a progress line, so append normally and reset progress tracking
            # self.last_progress_line = None
            # text = text + '\n'  # Add newline for regular prints

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)  # Insert the text at the end
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def keyPressEvent(self, event):        
        # Check if Ctrl+C was pressed
        if event.key() == Qt.Key_C and event.modifiers():
            self.close()
        else:
            super().keyPressEvent(event)

    def onWorkerFinished(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
