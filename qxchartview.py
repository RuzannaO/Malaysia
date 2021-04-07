from PyQt5.QtChart import QChartView
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import *

class QXChartView(QChartView):
    def __init__(self, parent = None):
        super(QChartView, self).__init__(parent)
        self.pos = None

       
    def drawForeground(self, painter, rect):
        if self.pos:
            pen = QPen(Qt.black, 1)
            painter.setPen(pen)
            painter.drawLine(self.pos[0],0,self.pos[0],250)


    # def mouseMoveEvent(self, event):
    #     super().mouseMoveEvent(event)
    #     self.pos = (event.pos().x(),event.pos().y())
    #     self.parent.update()

    # def mousePressEvent(self, event):
    #     super().mousePressEvent(event)
    #     self.pos = (event.pos().x(),event.pos().y())
    #     self.parent.update() 
    #     #print (event.globalX(),event.globalY()  )
    #     #print (event.pos().x(),event.pos().y()) 
