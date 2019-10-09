import math
import sys
import os
from os.path import join, isfile

import cv2 as cv
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, pyqtSlot 
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QPixmap, QImage
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
class GuiRoot(QWidget):
    # Root of Gui
    def __init__(self):
        super().__init__()     
        self.ui_init()

    def ui_init(self):
        # the application window size
        self.setFixedSize(1440, 960) 
        # Set image window size
        self.label_width = 1400
        self.label_height = 960
        self.image_label = QLabel() # label for image display
        self.image_label.resize(self.label_width, self.label_height)
        # build the GUI 
        self.file_gui()
        self.process_image_gui()
        self.setWindowTitle('HW 1-1')
        # put window in the center of monitor
        self.center()
        
        # gradient of img
        self.x = None
        self.y = None
        # the array for displaying current image
        self.img = np.ndarray(())
        # flag for comfirming if any image has been opened
        self.original_flag = False

        # control GUI layout
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.file_qgroupbox)
        vbox.addWidget(self.GS_qgroupbox)
        vbox.addWidget(self.sobel_qgroupbox)
        vbox.addWidget(self.ST_qgroupbox)
        vbox.addWidget(self.basic_qgroupbox)

        vbox.setStretchFactor(self.file_qgroupbox, 2)
        vbox.setStretchFactor(self.GS_qgroupbox, 3)
        vbox.setStretchFactor(self.sobel_qgroupbox, 3)
        vbox.setStretchFactor(self.ST_qgroupbox, 3)
        vbox.setStretchFactor(self.basic_qgroupbox, 3)

        vbox.insertSpacing(-1, 280)
        hbox.addLayout(vbox)
        hbox.addWidget(self.image_label)
        hbox.setStretchFactor(self.image_label, 3)
        self.setLayout(hbox)
        self.show()

    def file_gui(self):
        # finetune the datailed layout

        self.file_qgroupbox = QGroupBox("File: ")
        layout = QGridLayout()
        # open image related gui
        self.open_image_btn = QPushButton("Open Image", self)
        self.open_image_btn.clicked.connect(self.open_function)
        # save current image related gui
        self.save_image_btn = QPushButton("Save Image", self)
        self.save_image_btn.clicked.connect(self.save_function)
        layout.addWidget(self.open_image_btn, 0, 0, 1, 1)
        layout.addWidget(self.save_image_btn, 0, 1, 1, 1)
        layout.setVerticalSpacing(0)
        layout.setHorizontalSpacing(0)
        self.file_qgroupbox.setLayout(layout)
        
    def process_image_gui(self):
        # process image related gui

        # Gaussian Smooth related part
        self.GS_qgroupbox = QGroupBox("Gaussian Smooth: ")
        GS_layout = QGridLayout()
        GS_layout.setSpacing(50)
        self.GS_btn = QPushButton("Apply", self)
        self.GS_btn.clicked.connect(self.gaussian_smooth)
        self.GS_sigam_label = QLabel("Sigma of Gaussian")
        self.G_sigma = QDoubleSpinBox()
        self.G_sigma.setDecimals(3)
        self.G_sigma.setValue(5)
        self.G_sigma.setRange(0,10)
        self.GS_kernel_label = QLabel("Kernal Size")
        self.kernel_size = QDoubleSpinBox()
        self.kernel_size.setDecimals(0)
        self.kernel_size.setValue(3)
        GS_layout.addWidget(self.GS_btn, 2, 0, 1, 2)
        GS_layout.addWidget(self.GS_sigam_label, 0, 0, 1, 1)
        GS_layout.addWidget(self.G_sigma, 0, 1, 1, 1)
        GS_layout.addWidget(self.GS_kernel_label, 1, 0, 1, 1)
        GS_layout.addWidget(self.kernel_size, 1, 1, 1, 1)
        GS_layout.setVerticalSpacing(0)
        GS_layout.setHorizontalSpacing(0)
        self.GS_qgroupbox.setLayout(GS_layout)
        # Sobel edge detection related part
        self.sobel_qgroupbox = QGroupBox("Sobel Edge Detection: ")
        sobel_layout = QGridLayout()
        sobel_layout.setSpacing(50)
        self.sobel_btn = QPushButton("Apply", self)
        self.sobel_btn.clicked.connect(self.sobel_edge_detection)
        self.mag_dir_choose = QComboBox()
        self.mag_dir_choose.addItem("Magnitude of Gradient")
        self.mag_dir_choose.addItem("Direction of Gradient")
        self.sobel_threshold_label = QLabel("threshold")
        self.sobel_threshold = QDoubleSpinBox()
        self.sobel_threshold.setDecimals(0)
        self.sobel_threshold.setValue(30)
        self.sobel_threshold.setRange(0,255)
        sobel_layout.addWidget(self.sobel_btn, 2, 0, 1, 2)
        sobel_layout.addWidget(self.sobel_threshold_label, 0, 0, 1, 1)
        sobel_layout.addWidget(self.sobel_threshold, 0, 1, 1, 1)
        sobel_layout.addWidget(self.mag_dir_choose, 1, 0, 1, 2)
        sobel_layout.setVerticalSpacing(0)
        sobel_layout.setHorizontalSpacing(0)
        self.sobel_qgroupbox.setLayout(sobel_layout)
        # structure tensor related part
        self.ST_qgroupbox = QGroupBox("Structure Tensor: ")
        ST_layout = QGridLayout()
        ST_layout.setSpacing(50)
        self.structure_tensor_btn = QPushButton("Apply", self)
        self.structure_tensor_btn.clicked.connect(self.structure_tensor)
        self.structure_window_label = QLabel("window size of Harris")
        self.structure_window = QDoubleSpinBox()
        self.structure_window.setDecimals(0)
        self.structure_window.setValue(3)
        self.structure_window.setRange(2,100)
        self.nms_window_label = QLabel("window size of NMS")
        self.nms_window = QSpinBox()
        self.nms_window.setValue(3)
        self.nms_window.setRange(2,100)
        ST_layout.addWidget(self.structure_tensor_btn, 2, 0, 1, 2)
        ST_layout.addWidget(self.structure_window_label, 0, 0, 1, 1)
        ST_layout.addWidget(self.structure_window, 0, 1, 1, 1)
        ST_layout.addWidget(self.nms_window_label, 1, 0, 1, 1)
        ST_layout.addWidget(self.nms_window, 1, 1, 1, 1)
        ST_layout.setVerticalSpacing(0)
        ST_layout.setHorizontalSpacing(0)
        self.ST_qgroupbox.setLayout(ST_layout)
        #
        self.basic_qgroupbox = QGroupBox("Basic Process: ")
        basic_layout = QGridLayout()
        basic_layout.setSpacing(50)
        self.img_rotate_btn = QPushButton("Rotating", self)
        self.img_rotate_btn.clicked.connect(self.rotate)
        self.rotate_angle_label = QLabel("Angle:")
        self.rotate_angle = QDoubleSpinBox()
        self.rotate_angle.setDecimals(0)
        self.rotate_angle.setValue(30)
        self.rotate_angle.setRange(0, 360)
        basic_layout.addWidget(self.img_rotate_btn, 1, 0, 1, 2)
        basic_layout.addWidget(self.rotate_angle_label, 0, 0, 1, 1)
        basic_layout.addWidget(self.rotate_angle, 0, 1, 1, 1)
        #
        self.img_scaling_btn = QPushButton("Scaling", self)
        self.img_scaling_btn.clicked.connect(self.scale)
        self.scaling_ratio_label = QLabel("Ratio:")
        self.scaling_ratio = QDoubleSpinBox()
        self.scaling_ratio.setDecimals(3)
        self.scaling_ratio.setValue(0.5)
        self.scaling_ratio.setRange(0, 10)
        basic_layout.addWidget(self.img_scaling_btn, 3, 0, 1, 2)
        basic_layout.addWidget(self.scaling_ratio_label, 2, 0, 1, 1)
        basic_layout.addWidget(self.scaling_ratio, 2, 1, 1, 1)
        #
        self.Ori_btn = QPushButton("Original image", self)
        self.Ori_btn.clicked.connect(self.origin_function)
        basic_layout.addWidget(self.Ori_btn, 4, 0, 1, 2)
        basic_layout.setVerticalSpacing(0)
        basic_layout.setHorizontalSpacing(0)
        self.basic_qgroupbox.setLayout(basic_layout)
    def open_function(self):
        # open image function

        # the default location
        datapath = os.path.realpath(os.path.join(os.getcwd(), "ImageData"))
        # using dialog to open file
        file_name = QFileDialog.getOpenFileName(self, 'Open image', 
        datapath, "Image files(*.jpg *.png *.tif *.bmp *.raw)") # support image format
        if file_name[0]:
            # save the image by opencv
            self.img = cv.imread(file_name[0], -1)
            if len(self.img.shape) == 2:
                # if the image is gray scale
                self.img = np.stack((self.img,)*3, axis=-1)
            # save a copy of input image for recovering         
            self.original_img = np.copy(self.img)
            self.original_flag = True
            # display the image
            self.show_function(self.img)

    def show_function(self, img):
        # get the size and channel of image
        height, width, channel = img.shape
        # adjust the height and width for suiting the window size
        if height > self.label_height or width > self.label_width:
            ratio = max(height/self.label_height, width/self.label_width)
            img = cv.resize(img, (int(width/ratio), int(height/ratio)), interpolation=cv.INTER_CUBIC)
            height, width, channel = img.shape
        bytesPerLine = channel*width
        # turn image of opencv into Qimage
        qImg = QImage(img.data, width, height, bytesPerLine,
            QImage.Format_RGB888).rgbSwapped()
        # show Qimage on label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))

    def save_function(self):
        # save current image function
        file_name = QFileDialog.getSaveFileName(
            self, 'Save Image', './save_img', 'Image files(*.jpg *.png* .tif *.raw *.bmp)')
        if file_name[0]:
            cv.imwrite(file_name[0], self.img)

    def gaussian_smooth(self):
        def gaussian_2d(sigma, xx_yy):
            return np.exp(-xx_yy/(2*sigma**2))/(2*np.pi*sigma**2)
        def generate_gaussian_filter(kernel_size, sigma):
            g_filter = np.zeros((kernel_size, kernel_size))
            center = int(kernel_size/2)
            sumation = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    g_filter[i][j] = (j - center)**2 + (i - center)**2
                    g_filter[i][j] = gaussian_2d(sigma, g_filter[i][j])
                    sumation += g_filter[i][j]
            print(g_filter)
            return g_filter/sumation
        def convolve_function(g_filter, R, G, B):
            red = ndimage.convolve(R, g_filter, mode='nearest')
            green = ndimage.convolve(G, g_filter, mode='nearest')
            blue = ndimage.convolve(B, g_filter, mode='nearest')
            img = cv.merge((blue, green, red))
            return img
        if self.original_flag:
            (B,G,R) = cv.split(self.img)
            # height, width = B.shape
            sigma = self.G_sigma.value()
            kernel_size = int(self.kernel_size.value())
            gaussizn_filter = np.copy(generate_gaussian_filter(kernel_size, sigma))
            self.img = convolve_function(gaussizn_filter, R, G, B)
            self.show_function(self.img)
    def sobel_edge_detection(self):
        def sobel_filter_convolve(img):
            Fx = np.matrix([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            Fy = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            self.x = convolve2d(img, Fx, "same", "symm")
            self.y = convolve2d(img, Fy, "same", "symm")
            #self.x = np.where(self.x < threshold, 0, self.x)
            #self.y = np.where(self.y < threshold, 0, self.y)
            G = np.abs(self.x) + np.abs(self.y)# gradient magnitude
            D = np.arctan2(self.y, self.x)
            #D = D*(255/np.pi)+255
            D = D*127.5/np.pi+127.5
            print(np.max(D), np.min(D))
            #return self.x, self.y
            return (G-np.min(G))/(np.max(G)-np.min(G))*255, D
        if self.original_flag:
            threshold = int(self.sobel_threshold.value())
            img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)# double deal
            magnitude = sobel_filter_convolve(img)[0].astype(np.uint8())
            magnitude = np.where(magnitude < threshold, 0, magnitude)        
            if self.mag_dir_choose.currentText() == 'Magnitude of Gradient':
                self.img = np.stack((magnitude,)*3, axis=-1)
            else:
                direction = sobel_filter_convolve(img)[1].astype(np.uint8())
                direction = np.where(magnitude == 0, 0, direction)
                print(np.min(direction))
                self.img = cv.applyColorMap(direction, cv.COLORMAP_JET)
            
            self.show_function(self.img)
    def origin_function(self):
        # return the original image 
        if not self.original_flag:
            # haven't open any image file
            pass
        else:
            # reset img and show it
            self.img = np.copy(self.original_img)
            self.show_function(self.img)
    def structure_tensor(self):
        def nms(R, h, w, i, j, window_size):
            rad = int(window_size/2)
            radi = round(window_size/2)
            return (True if R[i,j] == np.max(R[max(0, i-rad) : min(i+radi, h-1), max(0, j-rad) : min(j+radi, w-1)]) else False)
        window_size = int(self.structure_window.value())
        nms_window_size = int(self.nms_window.value())
        window = np.ones((window_size, window_size))
        k = 0.04
        xx = convolve2d(self.x*self.x, window, "same", "fill")
        xy = convolve2d(self.x*self.y, window, "same", "fill")
        yy = convolve2d(self.y*self.y, window, "same", "fill")
        det = xx*yy - xy**2
        trace = xx + yy
        r = det - k*(trace**2)
        self.img = np.copy(self.original_img)
        height, width, channel = self.img.shape
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.img = np.stack((self.img,)*3, axis=-1)
        corner_def = np.percentile(r, 99)
        for h in range(height):
            for w in range(width):
                if r[h, w] > corner_def and nms(r, height, width, h, w, nms_window_size):
                    cv.circle(self.img, (w,h), 2, (0, 0, 255),-1)
                    # self.img[h,w,2]=255
        self.show_function(self.img)
    def center(self):
        # Place window in the center
        qr = self.frameGeometry()
        central_p = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(central_p)
        self.move(qr.topLeft())
    def rotate(self):
        angle = self.rotate_angle.value()
        self.img = ndimage.rotate(self.img, angle, reshape=True)
        self.show_function(self.img)
    def scale(self):
        ratio = self.scaling_ratio.value()
        height, width, channel = self.img.shape
        self.img = cv.resize(self.img, (int(width*ratio), int(height*ratio)), interpolation=cv.INTER_CUBIC)
        self.show_function(self.img)
if __name__ == '__main__':
    sys.argv += ['--style', 'fusion']
    app = QApplication(sys.argv)
    gui_root = GuiRoot() # instance of the gui and finction
    sys.exit(app.exec_())
'''
        def explicit_correlation(image, kernel):
            hi, wi= image.shape
            hk, wk = kernel.shape
            image_padded = np.zeros(shape=(hi + hk - 1, wi + wk - 1))    
            image_padded[hk//2:-hk//2, wk//2:-wk//2] = image
            out = np.zeros(shape=image.shape)
            for row in range(hi):
                for col in range(wi):
                    for i in range(hk):
                        for j in range(wk):
                            out[row, col] += image_padded[row + i, col + j]*kernel[i, j]
            return out
        #
                    for h in range(height):
                for w in range(width):
                    B[h, w]
'''