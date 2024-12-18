# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/main_frame.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(324, 373)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.videoprocess = QtWidgets.QLabel(self.centralwidget)
        self.videoprocess.setFrameShape(QtWidgets.QFrame.Box)
        self.videoprocess.setMidLineWidth(20)
        self.videoprocess.setText("")
        self.videoprocess.setAlignment(QtCore.Qt.AlignCenter)
        self.videoprocess.setObjectName("videoprocess")
        self.verticalLayout.addWidget(self.videoprocess)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.class_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Sitka Subheading Semibold")
        font.setBold(True)
        font.setWeight(75)
        self.class_label.setFont(font)
        self.class_label.setFrameShape(QtWidgets.QFrame.Box)
        self.class_label.setAlignment(QtCore.Qt.AlignCenter)
        self.class_label.setObjectName("class_label")
        self.horizontalLayout.addWidget(self.class_label)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.StartAttandance = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Noto Serif")
        self.StartAttandance.setFont(font)
        self.StartAttandance.setObjectName("StartAttandance")
        self.horizontalLayout.addWidget(self.StartAttandance)
        self.StopAttandance = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Noto Serif")
        self.StopAttandance.setFont(font)
        self.StopAttandance.setObjectName("StopAttandance")
        self.horizontalLayout.addWidget(self.StopAttandance)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.Log = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Noto Serif")
        self.Log.setFont(font)
        self.Log.setText("")
        self.Log.setAlignment(QtCore.Qt.AlignCenter)
        self.Log.setObjectName("Log")
        self.verticalLayout.addWidget(self.Log)
        self.verticalLayout.setStretch(0, 10)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 324, 18))
        self.menubar.setObjectName("menubar")
        self.menuReport = QtWidgets.QMenu(self.menubar)
        self.menuReport.setObjectName("menuReport")
        MainWindow.setMenuBar(self.menubar)
        self.actionAttandance_Report = QtWidgets.QAction(MainWindow)
        self.actionAttandance_Report.setObjectName("actionAttandance_Report")
        self.actionConfiguration = QtWidgets.QAction(MainWindow)
        self.actionConfiguration.setObjectName("actionConfiguration")
        self.actionAdd_Student = QtWidgets.QAction(MainWindow)
        self.actionAdd_Student.setObjectName("actionAdd_Student")
        self.menuReport.addAction(self.actionAttandance_Report)
        self.menuReport.addAction(self.actionConfiguration)
        self.menuReport.addAction(self.actionAdd_Student)
        self.menubar.addAction(self.menuReport.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.class_label.setText(_translate("MainWindow", "Standard"))
        self.StartAttandance.setText(_translate("MainWindow", "Start Attandance"))
        self.StopAttandance.setText(_translate("MainWindow", "Stop Attandance"))
        self.menuReport.setTitle(_translate("MainWindow", "Menu"))
        self.actionAttandance_Report.setText(_translate("MainWindow", "Attandance Report"))
        self.actionConfiguration.setText(_translate("MainWindow", "Configuration"))
        self.actionAdd_Student.setText(_translate("MainWindow", "Add Student"))
