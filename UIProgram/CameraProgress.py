# -*- coding: utf-8 -*-
# 摄像头保存弹窗信息
from PyQt5.QtWidgets import QDialog, QLabel,QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QCoreApplication

class CamProgress(QDialog):
    def __init__(self, parent=None):
        super(CamProgress, self).__init__(parent)

        self.resize(350, 100)
        self.setWindowTitle(self.tr("摄像头保存进度信息"))
        self.TipLabel = QLabel(self.tr("已保存 0 帧"))

        TipLayout = QHBoxLayout()
        TipLayout.addWidget(self.TipLabel)

        self.cancelButton = QPushButton('结束保存', self)
        buttonlayout = QHBoxLayout()
        buttonlayout.addStretch(1)
        buttonlayout.addWidget(self.cancelButton)

        layout = QVBoxLayout()
        layout.addLayout(TipLayout)
        layout.addLayout(buttonlayout)
        self.setLayout(layout)
        self.cancelButton.clicked.connect(self.onCancel)

    def setValue(self, start, fps):
        self.TipLabel.setText(self.tr("已保存{}帧,时长:{} s".format(start,round(start/fps,2))))

    def onCancel(self, event):
        self.close()
