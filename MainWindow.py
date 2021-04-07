# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'm.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")

        MainWindow.resize(1380, 1098)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.playlistView = QtWidgets.QListView(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.playlistView.sizePolicy().hasHeightForWidth())
        self.playlistView.setSizePolicy(sizePolicy)
        self.playlistView.setAcceptDrops(True)
        self.playlistView.setProperty("showDropIndicator", True)
        self.playlistView.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.playlistView.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.playlistView.setAlternatingRowColors(True)
        self.playlistView.setUniformItemSizes(True)
        self.playlistView.setObjectName("playlistView")
        self.horizontalLayout.addWidget(self.playlistView)
        self.live_label = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.live_label.sizePolicy().hasHeightForWidth())
        # self.live_label.setSizePolicy(sizePolicy)
        # self.live_label.setMinimumSize(QtCore.QSize(640, 480))
        # self.live_label.setMaximumSize(QtCore.QSize(640, 480))
        self.live_label.setText("")
        self.live_label.setObjectName("live_label")
        self.horizontalLayout.addWidget(self.live_label)
        self.Video = QVideoWidget(self.centralWidget)
        self.Video.setEnabled(True)
        self.Video.setObjectName("Video")
        self.horizontalLayout.addWidget(self.Video)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.Chart = QTimeLine(self.centralWidget,100)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Chart.sizePolicy().hasHeightForWidth())
        self.Chart.setSizePolicy(sizePolicy)
        self.Chart.setMinimumSize(QtCore.QSize(0, 200))
        self.Chart.setAutoFillBackground(False)
        self.Chart.setObjectName("Chart")
        self.verticalLayout.addWidget(self.Chart)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.previousEmotionState = QtWidgets.QPushButton(self.centralWidget)
        self.previousEmotionState.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/control-skip-180.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.previousEmotionState.setIcon(icon)
        self.previousEmotionState.setObjectName("previousEmotionState")
        self.horizontalLayout_5.addWidget(self.previousEmotionState)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox_2)
        self.nextEmotionState = QtWidgets.QPushButton(self.centralWidget)
        self.nextEmotionState.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/control-skip.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.nextEmotionState.setIcon(icon1)
        self.nextEmotionState.setObjectName("nextEmotionState")
        self.horizontalLayout_5.addWidget(self.nextEmotionState)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.currentTimeLabel = QtWidgets.QLabel(self.centralWidget)
        self.currentTimeLabel.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.currentTimeLabel.sizePolicy().hasHeightForWidth())
        self.currentTimeLabel.setSizePolicy(sizePolicy)
        self.currentTimeLabel.setMinimumSize(QtCore.QSize(30, 0))
        self.currentTimeLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.currentTimeLabel.setIndent(0)
        self.currentTimeLabel.setObjectName("currentTimeLabel")
        self.horizontalLayout_4.addWidget(self.currentTimeLabel)
        self.timeSlider = QtWidgets.QSlider(self.centralWidget)
        self.timeSlider.setStyleSheet("QSlider::handle:horizontal {\n"
"    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);\n"
"    border: 1px solid #5c5c5c;\n"
"    width: 2px;\n"
"    margin: -2px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */\n"
"    border-radius: 3px;\n"
"    background-image: url(\"images/control.png\"); \n"
"    background-repeat:no-repeat;\n"
"    background-position: center center;\n"
"}")
        self.timeSlider.setOrientation(QtCore.Qt.Horizontal)
        self.timeSlider.setObjectName("timeSlider")
        self.horizontalLayout_4.addWidget(self.timeSlider)
        self.totalTimeLabel = QtWidgets.QLabel(self.centralWidget)
        self.totalTimeLabel.setMinimumSize(QtCore.QSize(80, 0))
        self.totalTimeLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.totalTimeLabel.setObjectName("totalTimeLabel")
        self.horizontalLayout_4.addWidget(self.totalTimeLabel)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.previousButton = QtWidgets.QPushButton(self.centralWidget)
        self.previousButton.setText("")
        self.previousButton.setIcon(icon)
        self.previousButton.setObjectName("previousButton")
        self.horizontalLayout_3.addWidget(self.previousButton)

        self.nextButton = QtWidgets.QPushButton(self.centralWidget)
        self.nextButton.setText("")
        self.nextButton.setIcon(icon1)
        self.nextButton.setObjectName("nextButton")
        self.horizontalLayout_3.addWidget(self.nextButton)

        self.playButton = QtWidgets.QPushButton(self.centralWidget)
        self.playButton.setText("Play")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("images/control.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.playButton.setIcon(icon2)
        self.playButton.setObjectName("playButton")
        self.horizontalLayout_3.addWidget(self.playButton)

        self.pauseButton = QtWidgets.QPushButton(self.centralWidget)
        self.pauseButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("images/control-pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pauseButton.setIcon(icon3)
        self.pauseButton.setObjectName("pauseButton")
        self.horizontalLayout_3.addWidget(self.pauseButton)

        self.stopButton = QtWidgets.QPushButton(self.centralWidget)
        self.stopButton.setText('Stop')
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("images/control-stop-square.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopButton.setIcon(icon4)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_3.addWidget(self.stopButton)



        self.clear_video = QtWidgets.QPushButton(self.centralWidget)
        self.clear_video.setText("Close Video")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("images/close_video1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clear_video.setIcon(icon7)
        self.clear_video.setObjectName("clear_video")
        self.horizontalLayout_3.addWidget(self.clear_video)

        self.viewButton = QtWidgets.QPushButton(self.centralWidget)
        self.viewButton.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("images/control-live-square.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.viewButton.setIcon(icon5)
        self.viewButton.setCheckable(True)
        self.viewButton.setObjectName("viewButton")
        self.horizontalLayout_3.addWidget(self.viewButton)


        self.viewButton1 = QtWidgets.QPushButton(self.centralWidget)
        self.viewButton1.setText("Record")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("images/control-live-square-yellow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.viewButton1.setIcon(icon6)
        self.viewButton1.setCheckable(True)
        self.viewButton1.setObjectName("viewButton1")
        self.horizontalLayout_3.addWidget(self.viewButton1)




        spacerItem = QtWidgets.QSpacerItem(500, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.label_3 = QtWidgets.QLabel(self.centralWidget)
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("images/speaker-volume.png"))
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.volumeSlider = QtWidgets.QSlider(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.volumeSlider.sizePolicy().hasHeightForWidth())
        self.volumeSlider.setSizePolicy(sizePolicy)
        self.volumeSlider.setMinimumSize(QtCore.QSize(5, 0))
        self.volumeSlider.setMaximumSize(QtCore.QSize(8000, 16777215))
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setProperty("value", 100)
        self.volumeSlider.setOrientation(QtCore.Qt.Horizontal)
        self.volumeSlider.setObjectName("volumeSlider")
        self.horizontalLayout_3.addWidget(self.volumeSlider)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(2, 2)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1380, 41))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")


        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.open_file_action = QtWidgets.QAction(MainWindow)
        self.open_file_action.setObjectName("open_file_action")
        self.menuFile.addAction(self.open_file_action)

        self.open_folder_action = QtWidgets.QAction(MainWindow)
        self.open_folder_action.setObjectName("open_folder_action")
        self.menuFile.addAction(self.open_folder_action)

        self.clear_playlist_action = QtWidgets.QAction(MainWindow)
        self.clear_playlist_action.setObjectName("clear_playlist _action")
        self.menuFile.addAction(self.clear_playlist_action)

        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionExit)


        self.help_action = QtWidgets.QAction(MainWindow)
        self.menuHelp = QtWidgets.QMenu(self.menuBar)
        self.menuHelp.addAction(self.help_action)
        self.menuHelp.setObjectName("menuHelp")



        self.menuCamera = QtWidgets.QMenu(self.menuBar)
        self.menuCamera.setObjectName("menuCamera")
        # self.actionCamera = QtWidgets.QAction(MainWindow)
        # self.actionCamera.setObjectName("actionCamera")
        # self.menuCamera.addAction(self.actionCamera)

        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuCamera.menuAction())

        self.menuBar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Analyzer version 1.0"))
        self.label_2.setText(_translate("MainWindow", "Playlist"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "EMOTIONS"))
        self.currentTimeLabel.setText(_translate("MainWindow", "0:00"))
        self.totalTimeLabel.setText(_translate("MainWindow", "0:00"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))

        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.help_action.setText(_translate("MainWindow", "Instructions"))


        self.menuCamera.setTitle(_translate("MainWindow", "Camera Devices"))
        # self.actionCamera.setText(_translate("MainWindow", "Device 1"))

        self.open_file_action.setText(_translate("MainWindow", "Open File"))
        self.open_folder_action.setText(_translate("MainWindow", "Open Folder"))
        self.clear_playlist_action.setText(_translate("MainWindow", "Clear Playlist"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))




from PyQt5.QtMultimediaWidgets import QVideoWidget
from qtimeline import QTimeLine
