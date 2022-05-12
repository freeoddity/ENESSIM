
import sys
from PyQt5.QtGui import QIcon, QPainter, QColor, QPen, qRgb
from PyQt5.QtCore import QCoreApplication, QRect, QSize, Qt, QRectF
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGraphicsScene, QGraphicsView,QGridLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QIcon, QPixmap, QFont, QFontMetrics

class Map(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        self.setMinimumSize(500, 500) # min-size of the widget
        self.columns = 50 # num of columns in grid
        self.rows = 50 # num of rows in grid

        grid = QGridLayout(self)
        self.view = QGraphicsView()
        grid.addWidget(self.view)
        self.view.setRenderHints(QPainter.Antialiasing)

        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        brush = QBrush(QColor(qRgb(0, 0, 255))) # background color of square
        pen = QPen(Qt.black) # border color of square
        side = 10
        rect = QRectF(0, 0, side, side)
        for i in range(self.rows):
            for j in range(self.columns):
                self.scene.addRect(rect.translated(i * side, j * side), pen, brush)

        # this is required to ensure that fitInView works on first shown too
        self.resizeScene()

    def resizeScene(self):
        self.view.fitInView(self.scene.sceneRect())

    def resizeEvent(self, event):
        # call fitInView each time the widget is resized
        self.resizeScene()

    def showEvent(self, event):
        # call fitInView each time the widget is shown
        self.resizeScene()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Map()
    ex.show()
    sys.exit(app.exec_())