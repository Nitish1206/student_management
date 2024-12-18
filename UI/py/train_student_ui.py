# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/train_student.ui'
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
        self.trainingvideo = QtWidgets.QLabel(self.centralwidget)
        self.trainingvideo.setFrameShape(QtWidgets.QFrame.Box)
        self.trainingvideo.setMidLineWidth(20)
        self.trainingvideo.setText("")
        self.trainingvideo.setAlignment(QtCore.Qt.AlignCenter)
        self.trainingvideo.setObjectName("trainingvideo")
        self.verticalLayout.addWidget(self.trainingvideo)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.studentNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.studentNameLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.studentNameLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.studentNameLabel.setObjectName("studentNameLabel")
        self.horizontalLayout_2.addWidget(self.studentNameLabel)
        self.StudentNameEditor = QtWidgets.QLineEdit(self.centralwidget)
        self.StudentNameEditor.setAlignment(QtCore.Qt.AlignCenter)
        self.StudentNameEditor.setObjectName("StudentNameEditor")
        self.horizontalLayout_2.addWidget(self.StudentNameEditor)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.StudentStandardLabel = QtWidgets.QLabel(self.centralwidget)
        self.StudentStandardLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.StudentStandardLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.StudentStandardLabel.setObjectName("StudentStandardLabel")
        self.horizontalLayout.addWidget(self.StudentStandardLabel)
        self.standardEditor = QtWidgets.QLineEdit(self.centralwidget)
        self.standardEditor.setAlignment(QtCore.Qt.AlignCenter)
        self.standardEditor.setObjectName("standardEditor")
        self.horizontalLayout.addWidget(self.standardEditor)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 10)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.trainstudentbutton = QtWidgets.QPushButton(self.centralwidget)
        self.trainstudentbutton.setObjectName("trainstudentbutton")
        self.gridLayout.addWidget(self.trainstudentbutton, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAttandance_Report = QtWidgets.QAction(MainWindow)
        self.actionAttandance_Report.setObjectName("actionAttandance_Report")
        self.actionConfiguration = QtWidgets.QAction(MainWindow)
        self.actionConfiguration.setObjectName("actionConfiguration")
        self.actionAdd_Student = QtWidgets.QAction(MainWindow)
        self.actionAdd_Student.setObjectName("actionAdd_Student")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.studentNameLabel.setText(_translate("MainWindow", "Student ID"))
        self.StudentStandardLabel.setText(_translate("MainWindow", "Standard"))
        self.trainstudentbutton.setText(_translate("MainWindow", "Add Student"))
        self.actionAttandance_Report.setText(_translate("MainWindow", "Attandance Report"))
        self.actionConfiguration.setText(_translate("MainWindow", "Configuration"))
        self.actionAdd_Student.setText(_translate("MainWindow", "Add Student"))
