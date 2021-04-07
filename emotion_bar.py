from PyQt5.QtChart import QChart, QChartView, QHorizontalStackedBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
from PyQt5.Qt import Qt
from PyQt5.QtGui import QPainter

def configureChart(chartView):

    set0 = QBarSet('Emotion1')
    set1 = QBarSet('Emotion2')
    set2 = QBarSet('Emotion3')
    set3 = QBarSet('Emotion4')
    set4 = QBarSet('Emotion5')
    set5 = QBarSet('Emotion5')


    set0.append([100])
    set1.append([100])
    set0.append([50])


    series = QHorizontalStackedBarSeries()
    series.append(set0)
    series.append(set1)
    series.append(set0)


    chart = QChart()
    chart.addSeries(series)
    #chart.setTitle('Horizontal Bar Chart Demo')

    chart.setAnimationOptions(QChart.SeriesAnimations)

    months = ('Em1')

    axisY = QBarCategoryAxis()
    axisY.append(months)
    chart.addAxis(axisY, Qt.AlignLeft)
    series.attachAxis(axisY)

    axisX = QValueAxis()
    chart.addAxis(axisX, Qt.AlignBottom)
    series.attachAxis(axisX)

    axisX.applyNiceNumbers()

    chart.legend().setVisible(True)
    chart.legend().setAlignment(Qt.AlignBottom)
    chart.layout().setContentsMargins(-50, -50, -50, -50)
    chart.setBackgroundRoundness(0)
    chartView.setRenderHint(QPainter.Antialiasing)
    chartView.setChart(chart)
