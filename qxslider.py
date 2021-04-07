from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QSlider,QVBoxLayout)
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt


class QXSlider(QSlider):
    def __init__(self, chartWidget):
        super(QXSlider, self).__init__()
        self.chartWidget = chartWidget

    def mouseMoveEvent(self, event):
        print ('mm')
        self.chartWidget.pos = (event.pos().x(),event.pos().y())
        self.chartWidget.viewport().update()
        super().mouseMoveEvent(event)
  
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.chartWidget.viewport().update()
        self.chartWidget.pos = (event.pos().x(),event.pos().y())

        #print (event.globalX(),event.globalY()  )
        #print (event.pos().x(),event.pos().y()) 
