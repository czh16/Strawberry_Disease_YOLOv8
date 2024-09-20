# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication , QMainWindow, QFileDialog, \
    QMessageBox,QWidget,QHeaderView,QTableWidgetItem, QAbstractItemView
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
import sys
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal,QCoreApplication
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
from UIProgram.CameraProgress import CamProgress
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

class MainWindow(QMainWindow):
    update_camera_ui_signal = pyqtSignal(int)
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()

        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

        self.conf = 0.45
        self.iou = 0.7

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_result)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)
        self.ui.doubleSpinBox.valueChanged.connect(self.conf_value_change)
        self.ui.doubleSpinBox_2.valueChanged.connect(self.iou_value_change)
        self.ui.tableWidget.cellClicked.connect(self.on_cell_clicked)
        self.update_camera_ui_signal.connect(self.update_camera_process)

    def initMain(self):
        self.show_width = 610
        self.show_height = 380

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        self.save_camera_flag = False #摄像头保存标志

        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO(Config.model_path, task='segment')
        self.model(np.zeros((48, 48, 3)), device=self.device)  #预先加载推理模型
        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        # self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        # 表格
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.ui.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 80)
        self.ui.tableWidget.setColumnWidth(3, 150)
        self.ui.tableWidget.setColumnWidth(4, 90)
        self.ui.tableWidget.setColumnWidth(5, 230)
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 表格铺满
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        # self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

        # 设置主页背景图片border-image: url(:/icons/ui_imgs/icons/camera.png)
        # self.setStyleSheet("#MainWindow{background-image:url(:/bgs/ui_imgs/bg3.jpg)}")

    def conf_value_change(self):
        # 改变置信度值
        cur_conf = round(self.ui.doubleSpinBox.value(),2)
        self.conf = cur_conf

    def iou_value_change(self):
        # 改变iou值
        cur_iou = round(self.ui.doubleSpinBox_2.value(),2)
        self.iou = cur_iou

    def open_img(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.cap = None
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.VideoBtn.setText('打开视频')

        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开.jpg与.gif结尾的图片文件
        # file_path, _ = QFileDialog.getOpenFileName(self.ui.centralwidget, '打开图片', './', "Image files (*.jpg *.gif)")
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 目标检测
        t1 = time.time()
        self.results = self.model(self.org_path, conf=self.conf, iou=self.iou)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]
        self.id_list = [i for i in range(len(self.location_list))]

        # 原始图像高和宽
        img_height, img_width = self.results.orig_shape
        total_px = img_height * img_width  # 图片总像素点
        self.seg_percent_list = []
        if self.results.masks is None:
            # 计算总分割面积百分比
            seg_area_px = '0 %'
        else:
            # 计算总分割面积百分比
            seg_area_px = self.cal_seg_percent(self.results.masks.xy, total_px)
            for each_seg in self.results.masks.xy:
                # print(each_seg)
                # 计算每个分割目标面积百分比
                seg_percent = self.cal_seg_percent([each_seg], total_px)
                self.seg_percent_list.append(seg_percent)
        # print(self.seg_percent_list)

        # 自己绘制框与标签
        # now_img = self.cv_img.copy()
        # for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
        #     type_id = int(type_id)
        #     color = self.colors(int(type_id), True)
        #     # cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), colors(int(type_id), True), 3)
        #     now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)

        # 绘制窗口1图片
        draw_seg = True if self.ui.checkBox.isChecked() else False
        draw_box = True if self.ui.checkBox_2.isChecked() else False
        now_img = self.results.plot(boxes=draw_box,masks=draw_seg)

        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img,(self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 绘制窗口2图片
        win2_img = self.draw_seg_mask(self.results)
        resize_cvimg = cv2.resize(win2_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show_2.setPixmap(pix_img)
        self.ui.label_show_2.setAlignment(Qt.AlignCenter)
        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        self.ui.label_total_area_per.setText((str(seg_area_px)))

        # 设置目标选择下拉框
        choose_list = ['All']
        target_names = [Config.CH_names[id]+ '_'+ str(index) for index,id in enumerate(self.cls_list)]
        # object_list = sorted(set(self.cls_list))
        # for each in object_list:
        #     choose_list.append(Config.CH_names[each])
        choose_list = choose_list + target_names

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
        #   默认显示第一个目标框坐标
        #   设置坐标位置值
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
            self.ui.label_area_per.setText(str(self.seg_percent_list[0]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')
            self.ui.label_area_per.setText('')

        # # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list,self.seg_percent_list,self.id_list,path=self.org_path)

    def cal_seg_percent(self, mask_points, total_px):
        # 计算分割区域占图片总面积的像素百分比
        # 如果有尺寸参考，后期可用于面积计算
        area = 0
        for each_contour in mask_points:
            contour_int = each_contour.astype(np.int32)
            area += cv2.contourArea(contour_int)
        percent = '%.2f %%' % (area / total_px * 100)
        return percent


    def draw_seg_mask(self, result):
        # 绘制窗口2的图片:QImage需要uint8类型的数据。
        # 检测到结果时
        height, width = result.orig_shape
        if result.masks is not None and len(result.masks) > 0:
            mask_res = result.masks[0].cpu().data.numpy().transpose(1, 2, 0)
            for mask in result.masks[1:]:
                mask_res += mask.cpu().data.numpy().transpose(1, 2, 0)
            # res[res > 1] = 1
            # 单通道变3通道,并变为np.uint8类型
            mask_res = np.repeat(mask_res, 3, axis=2) * 255
            mask_res = mask_res.astype(np.uint8)
            mask_res = cv2.resize(mask_res, (width, height))
        else:
            # 检测不到结果时
            # 全黑的图像为PNG文件
            mask_res = np.zeros((height, width, 3), dtype=np.uint8)
        if self.ui.radioButton.isChecked():
            # 显示mask
            return mask_res
        else:
            # 显示mask对应的原始图片
            org_seg_img = cv2.bitwise_and(result.orig_img, mask_res)
            # org_seg_img[org_seg_img == 0] = 255 #背景设置白色
            return org_seg_img


    def detact_batch_imgs(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.cap = None
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.VideoBtn.setText('打开视频')

        directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if not  directory:
            return
        self.org_path = directory
        img_suffix = ['jpg','png','jpeg','bmp']
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory,file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                # self.ui.comboBox.setDisabled(False)
                img_path = full_path
                self.org_img = tools.img_cvread(img_path)
                # 目标检测
                t1 = time.time()
                self.results = self.model(img_path,conf=self.conf, iou=self.iou)[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
                self.id_list = [i for i in range(len(self.location_list))]

                # 原始图像高和宽
                img_height, img_width = self.results.orig_shape
                total_px = img_height * img_width  # 图片总像素点

                self.seg_percent_list = []
                if self.results.masks is None:
                    # 计算总分割面积百分比
                    seg_area_px = '0 %'
                else:
                    # 计算总分割面积百分比
                    seg_area_px = self.cal_seg_percent(self.results.masks.xy, total_px)
                    for each_seg in self.results.masks.xy:
                        # print(each_seg)
                        # 计算每个分割目标面积百分比
                        seg_percent = self.cal_seg_percent([each_seg], total_px)
                        self.seg_percent_list.append(seg_percent)

                draw_seg = True if self.ui.checkBox.isChecked() else False
                draw_box = True if self.ui.checkBox_2.isChecked() else False
                now_img = self.results.plot(boxes=draw_box,masks=draw_seg)

                self.draw_img = now_img
                # 获取缩放后的图片尺寸
                self.img_width, self.img_height = self.get_resize_size(now_img)
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)

                # 绘制窗口2图片
                win2_img = self.draw_seg_mask(self.results)
                resize_cvimg = cv2.resize(win2_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show_2.setPixmap(pix_img)
                self.ui.label_show_2.setAlignment(Qt.AlignCenter)

                # 目标数目
                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))
                self.ui.label_total_area_per.setText((str(seg_area_px)))

                # 设置目标选择下拉框
                choose_list = ['全部']
                target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
                choose_list = choose_list + target_names

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)

                if target_nums >= 1:
                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                    self.ui.label_conf.setText(str(self.conf_list[0]))
                    #   默认显示第一个目标框坐标
                    #   设置坐标位置值
                    self.ui.label_xmin.setText(str(self.location_list[0][0]))
                    self.ui.label_ymin.setText(str(self.location_list[0][1]))
                    self.ui.label_xmax.setText(str(self.location_list[0][2]))
                    self.ui.label_ymax.setText(str(self.location_list[0][3]))
                    self.ui.label_area_per.setText(str(self.seg_percent_list[0]))
                else:
                    self.ui.type_lb.setText('')
                    self.ui.label_conf.setText('')
                    self.ui.label_xmin.setText('')
                    self.ui.label_ymin.setText('')
                    self.ui.label_xmax.setText('')
                    self.ui.label_ymax.setText('')
                    self.ui.label_area_per.setText('')

                # # 删除表格所有行
                # self.ui.tableWidget.setRowCount(0)
                # self.ui.tableWidget.clearContents()
                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list,self.seg_percent_list, self.id_list, path=img_path)
                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()  #刷新页面

    def draw_rect_and_tabel(self, results, img):
        now_img = img.copy()
        location_list = results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
            type_id = int(type_id)
            color = self.colors(int(type_id), True)
            # cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), colors(int(type_id), True), 3)
            now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)

        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
        return now_img

    def combox_change(self):
        # 点击下拉框时触发，显示指定检测框结果
        com_text = self.ui.comboBox.currentText()
        if com_text == '全部':
            cur_box = self.location_list
            draw_seg = True if self.ui.checkBox.isChecked() else False
            draw_box = True if self.ui.checkBox_2.isChecked() else False
            # 窗口1图片
            cur_img = self.results.plot(boxes=draw_box,masks=draw_seg)
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
            self.ui.label_area_per.setText(str(self.seg_percent_list[0]))

            # 窗口2图片
            win2_img = self.draw_seg_mask(self.results)
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            draw_seg = True if self.ui.checkBox.isChecked() else False
            draw_box = True if self.ui.checkBox_2.isChecked() else False
            # 窗口1图片
            cur_img = self.results[index].plot(boxes=draw_box,masks=draw_seg)
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[index]])
            self.ui.label_conf.setText(str(self.conf_list[index]))
            self.ui.label_area_per.setText(str(self.seg_percent_list[index]))
            # 窗口2图片
            win2_img = self.draw_seg_mask(self.results[index])

        # 设置坐标位置值
        self.ui.label_xmin.setText(str(cur_box[0][0]))
        self.ui.label_ymin.setText(str(cur_box[0][1]))
        self.ui.label_xmax.setText(str(cur_box[0][2]))
        self.ui.label_ymax.setText(str(cur_box[0][3]))

        resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.clear()
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        resize_cvimg = cv2.resize(win2_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show_2.clear()
        self.ui.label_show_2.setPixmap(pix_img)
        self.ui.label_show_2.setAlignment(Qt.AlignCenter)


    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 清空下拉框
        self.ui.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(10)
        self.timer_camera.timeout.connect(self.open_frame)


    def tabel_info_show(self, locations, clses, confs, seg_percents, target_ids, path=None):
        path = path
        for location, cls, conf, seg_percent, target_id in zip(locations, clses, confs, seg_percents, target_ids):
            row_count = self.ui.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.ui.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count+1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            # item_path.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_location = QTableWidgetItem(str(location)) # 目标框位置
            # item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_seg_percent = QTableWidgetItem(str(seg_percent))
            item_seg_percent.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_target_id = QTableWidgetItem(str(target_id))
            item_target_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_target_id)
            self.ui.tableWidget.setItem(row_count, 3, item_cls)
            self.ui.tableWidget.setItem(row_count, 4, item_conf)
            self.ui.tableWidget.setItem(row_count, 5, item_location)
            self.ui.tableWidget.setItem(row_count, 6, item_seg_percent)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        # self.timer_info.stop()

    def open_frame(self):
        if self.cap is None:
            return
        ret, now_img = self.cap.read()
        if ret:
            # 目标检测
            t1 = time.time()
            results = self.model(now_img,conf=self.conf, iou=self.iou)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
            self.id_list = [i for i in range(len(self.location_list))]

            # 原始图像高和宽
            img_height, img_width = results.orig_shape
            total_px = img_height * img_width  # 图片总像素点

            self.seg_percent_list = []
            if results.masks is None:
                # 计算总分割面积百分比
                seg_area_px = '0 %'
            else:
                # 计算总分割面积百分比
                seg_area_px = self.cal_seg_percent(results.masks.xy, total_px)
                for each_seg in results.masks.xy:
                    # print(each_seg)
                    # 计算每个分割目标面积百分比
                    seg_percent = self.cal_seg_percent([each_seg], total_px)
                    self.seg_percent_list.append(seg_percent)

            # 绘制窗口1图片
            draw_seg = True if self.ui.checkBox.isChecked() else False
            draw_box = True if self.ui.checkBox_2.isChecked() else False
            now_img = results.plot(boxes=draw_box, masks=draw_seg)

            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 绘制窗口2图片
            win2_img = self.draw_seg_mask(results)
            resize_cvimg = cv2.resize(win2_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show_2.setPixmap(pix_img)
            self.ui.label_show_2.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))
            self.ui.label_total_area_per.setText((str(seg_area_px)))

            # 设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            # object_list = sorted(set(self.cls_list))
            # for each in object_list:
            #     choose_list.append(Config.CH_names[each])
            choose_list = choose_list + target_names

            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(choose_list)

            if target_nums >= 1:
                self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.ui.label_conf.setText(str(self.conf_list[0]))
                #   默认显示第一个目标框坐标
                #   设置坐标位置值
                self.ui.label_xmin.setText(str(self.location_list[0][0]))
                self.ui.label_ymin.setText(str(self.location_list[0][1]))
                self.ui.label_xmax.setText(str(self.location_list[0][2]))
                self.ui.label_ymax.setText(str(self.location_list[0][3]))
                self.ui.label_area_per.setText(str(self.seg_percent_list[0]))
            else:
                self.ui.type_lb.setText('')
                self.ui.label_conf.setText('')
                self.ui.label_xmin.setText('')
                self.ui.label_ymin.setText('')
                self.ui.label_xmax.setText('')
                self.ui.label_ymax.setText('')
                self.ui.label_area_per.setText('')

            # 删除表格所有行
            # self.ui.tableWidget.setRowCount(0)
            # self.ui.tableWidget.clearContents()
            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list,self.seg_percent_list, self.id_list, path=self.org_path)

            if self.save_camera_flag is True:
                self.count_nums += 1
                self.CameraSave(results, self.count_nums,fps=30)
        else:
            self.cap.release()
            self.timer_camera.stop()
            self.ui.VideoBtn.setText('打开视频')


    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            if self.cap and self.cap.isOpened():
                self.cap.release()
                cv2.destroyAllWindows()

        if self.cap and self.cap.isOpened():
            # 关闭视频
            self.ui.VideoBtn.setText('打开视频')
            self.ui.label_show.setText('')
            self.ui.label_show_2.setText('')
            self.cap.release()
            cv2.destroyAllWindows()
            self.ui.label_show.clear()
            self.ui.label_show_2.clear()
            return

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.ui.VideoBtn.setText('关闭视频')
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.VideoBtn.setText('打开视频')
            self.ui.CapBtn.setText('关闭摄像头')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.ui.comboBox.setDisabled(True)
        else:
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.label_show.setText('')
            self.ui.label_show_2.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.cap =None
            self.ui.label_show.clear()
            self.ui.label_show_2.clear()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width , depth= _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_result(self):
        # 保存图片，视频及摄像头检测结果
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            # 保存摄像头检测结果
            res = QMessageBox.information(self, '提示', '摄像头检测结果存储, 请确认是否开始保存？', QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.Yes)
            if res == QMessageBox.Yes:
                # 初始化存储结果
                self.save_camera_flag = True
                self.count_nums = 0
                fps=30
                width = int(self.cap.get(3))
                height = int(self.cap.get(4))
                size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 为视频编码方式，保存为avi格式
                save_camera_path = os.path.join(Config.save_path, 'camera.avi')
                mask_save_camera_path = os.path.join(Config.save_path, 'camera_mask.avi')
                seg_save_camera_path = os.path.join(Config.save_path, 'camera_seg.avi')
                self.out = cv2.VideoWriter(save_camera_path, fourcc, fps, size)
                self.mask_out = cv2.VideoWriter(mask_save_camera_path, fourcc, fps, size)
                self.seg_out = cv2.VideoWriter(seg_save_camera_path, fourcc, fps, size)
            else:
                return
            return

        if self.cap:
            # 保存视频
            res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                self.ui.VideoBtn.setText('打开视频')
                com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text,self.conf,self.iou)
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
            # 保存图片
            if os.path.isfile(self.org_path) and self.org_path.split('.')[-1].lower() in img_suffix:
                fileName = os.path.basename(self.org_path)
                name , end_name= fileName.rsplit(".",1)
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                save_mask_name = name + '_detect_result_mask.' + end_name
                save_img_mask_path = os.path.join(Config.save_path, save_mask_name)
                save_seg_name = name + '_detect_result_seg.' + end_name
                save_img_seg_path = os.path.join(Config.save_path, save_seg_name)
                results = self.model(self.org_path, conf=self.conf, iou=self.iou)[0]
                now_img = results.plot()
                img_mask_res, img_seg_res = self.get_mask_and_seg(results)

                # 保存图片
                cv2.imwrite(save_img_path, now_img)
                cv2.imwrite(save_img_mask_path, img_mask_res)
                cv2.imwrite(save_img_seg_path, img_seg_res)
                QMessageBox.about(self, '提示', '图片保存成功!\n存储目录:{}'.format(Config.save_path))
            elif os.path.isdir(self.org_path):
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.rsplit(".",1)
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        save_mask_name = name + '_detect_result_mask.' + end_name
                        save_img_mask_path = os.path.join(Config.save_path, save_mask_name)
                        save_seg_name = name + '_detect_result_seg.' + end_name
                        save_img_seg_path = os.path.join(Config.save_path, save_seg_name)
                        results = self.model(full_path,conf=self.conf, iou=self.iou)[0]
                        now_img = results.plot()
                        img_mask_res, img_seg_res = self.get_mask_and_seg(results)

                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)
                        cv2.imwrite(save_img_mask_path, img_mask_res)
                        cv2.imwrite(save_img_seg_path, img_seg_res)
                QMessageBox.about(self, '提示', '图片保存成功!\n存储目录:{}'.format(Config.save_path))


    def get_mask_and_seg(self, result):
        # 获取图片分割的mask和分割后的结果
        height, width = result.orig_shape
        if result.masks is not None and len(result.masks) > 0:
            mask_res = result.masks[0].cpu().data.numpy().transpose(1, 2, 0)
            for mask in result.masks[1:]:
                mask_res += mask.cpu().data.numpy().transpose(1, 2, 0)
            # res[res > 1] = 1
            # 单通道变3通道,并变为np.uint8类型
            mask_res = np.repeat(mask_res, 3, axis=2) * 255
            mask_res = mask_res.astype(np.uint8)
            mask_res = cv2.resize(mask_res, (width, height))
        else:
            # 检测不到结果时
            # 全黑的图像为PNG文件
            mask_res = np.zeros((height, width, 3), dtype=np.uint8)
        # 显示mask对应的原始图片
        org_seg_img = cv2.bitwise_and(result.orig_img, mask_res)
        # org_seg_img[org_seg_img == 0] = 255 #背景设置白色
        return mask_res, org_seg_img

    def update_process_bar(self,cur_num, total):
        # 更新保存的进度条
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total *100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    def update_camera_process(self,cur_num,fps=30):
        # 更新摄像头存储结果的弹窗信息
        if cur_num == 1:
            self.progress = CamProgress(self)
            self.progress.show()
        if self.progress.isVisible() is False:
            # 点击取消保存时，终止进程
            self.save_camera_flag = False
            self.out.release()
            self.mask_out.release()
            self.seg_out.release()
            self.progress.close()
            return
        self.progress.setValue(cur_num, fps)
        QApplication.processEvents()

    def on_cell_clicked(self, row, column):
        """
        鼠标点击表格触发，界面显示当前行内容
        """
        if self.cap:
            # 视频或摄像头不支持表格选择
            return
        img_path = self.ui.tableWidget.item(row, 1).text()
        target_id = int(self.ui.tableWidget.item(row, 2).text())
        now_type = self.ui.tableWidget.item(row, 3).text()
        conf_value = self.ui.tableWidget.item(row, 4).text()
        location_value = eval(self.ui.tableWidget.item(row, 5).text())
        seg_area_percent = self.ui.tableWidget.item(row, 6).text()

        self.ui.type_lb.setText(now_type)
        self.ui.label_conf.setText(str(conf_value))
        self.ui.label_xmin.setText(str(location_value[0]))
        self.ui.label_ymin.setText(str(location_value[1]))
        self.ui.label_xmax.setText(str(location_value[2]))
        self.ui.label_ymax.setText(str(location_value[3]))
        self.ui.label_area_per.setText(str(seg_area_percent))

        cur_commbox_text = now_type + '_' + str(target_id)

        now_img = tools.img_cvread(img_path)
        # 目标检测
        t1 = time.time()
        self.results = self.model(now_img, conf=self.conf, iou=self.iou)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        # 原始图像高和宽
        img_height, img_width = self.results.orig_shape
        total_px = img_height * img_width  # 图片总像素点
        self.seg_percent_list = []
        if self.results.masks is None:
            # 计算总分割面积百分比
            seg_area_px = '0 %'
        else:
            # 计算总分割面积百分比
            seg_area_px = self.cal_seg_percent(self.results.masks.xy, total_px)
            for each_seg in self.results.masks.xy:
                # print(each_seg)
                # 计算每个分割目标面积百分比
                seg_percent = self.cal_seg_percent([each_seg], total_px)
                self.seg_percent_list.append(seg_percent)

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)
        self.ui.comboBox.setCurrentText(cur_commbox_text)

        # 绘制窗口1图片
        draw_seg = True if self.ui.checkBox.isChecked() else False
        draw_box = True if self.ui.checkBox_2.isChecked() else False
        now_img = self.results[target_id].plot(boxes=draw_box, masks=draw_seg)
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 绘制窗口2图片
        win2_img = self.draw_seg_mask(self.results[target_id])
        resize_cvimg = cv2.resize(win2_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show_2.setPixmap(pix_img)
        self.ui.label_show_2.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        self.ui.label_total_area_per.setText((str(seg_area_px)))

    def CameraSave(self, results, nums, fps=30):
        # 摄像头检测结果存储
        print('已存储{}帧，时长{}s'.format(nums, round(nums / fps, 2)))
        frame = results.plot()
        img_mask_res, img_seg_res = self.get_mask_and_seg(results)
        self.out.write(frame)
        self.mask_out.write(img_mask_res)
        self.seg_out.write(img_seg_res)
        self.update_camera_ui_signal.emit(nums)


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int,int)

    def __init__(self, path, model, com_text,conf,iou):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        self.conf = conf
        self.iou = iou
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式“xvid”
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        mask_save_name = name + '_detect_result_mask.avi'
        mask_save_video_path = os.path.join(Config.save_path, mask_save_name)
        mask_out = cv2.VideoWriter(mask_save_video_path, fourcc, fps, size)

        seg_save_name = name + '_detect_result_seg.avi'
        seg_save_video_path = os.path.join(Config.save_path, seg_save_name)
        seg_out = cv2.VideoWriter(seg_save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret == True:
                # 检测
                results = self.model(frame,conf=self.conf,iou=self.iou)[0]
                frame = results.plot()
                img_mask_res, img_seg_res = self.get_mask_and_seg(results)
                out.write(frame)
                mask_out.write(img_mask_res)
                seg_out.write(img_seg_res)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()
        mask_out.release()
        seg_out.release()

    def get_mask_and_seg(self, result):
        # 获取图片分割的mask和分割后的结果
        height, width = result.orig_shape
        if result.masks is not None and len(result.masks) > 0:
            mask_res = result.masks[0].cpu().data.numpy().transpose(1, 2, 0)
            for mask in result.masks[1:]:
                mask_res += mask.cpu().data.numpy().transpose(1, 2, 0)
            # res[res > 1] = 1
            # 单通道变3通道,并变为np.uint8类型
            mask_res = np.repeat(mask_res, 3, axis=2) * 255
            mask_res = mask_res.astype(np.uint8)
            mask_res = cv2.resize(mask_res, (width, height))
        else:
            # 检测不到结果时
            # 全黑的图像为PNG文件
            mask_res = np.zeros((height, width, 3), dtype=np.uint8)
        # 显示mask对应的原始图片
        org_seg_img = cv2.bitwise_and(result.orig_img, mask_res)
        # org_seg_img[org_seg_img == 0] = 255 #背景设置白色
        return mask_res, org_seg_img

    def stop(self):
        # 停止保存
        self.is_running = False



if __name__ == "__main__":
    # 对于按钮文字显示不全的，完成高清屏幕自适应设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
