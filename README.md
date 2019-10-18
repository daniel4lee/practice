# Homework 1 of Computer Vision course

## Preview

![preview](https://imgur.com/3oNLt7U.gif)

## Introduction

![intro pic](https://imgur.com/QX2ctCa.jpg)

1. __Open Image:__ Clicking this button to load image files. After selecting a image file, it would be displayed at the right side of window, #17. _Note that if the image size is larger than (1400,960), it would be scaled to smaller size. But remaining the same proportion._
2. __Save Image:__ The button provide the function to save current displaying image into selection path.
3. __Angle:__ the angle of rotation.
4. __Rotating:__ Turn the image Anticlockwise rotation with the angle of #12.
5. __Ratio:__ the degree of zoom out.
6. __Scaling:__ Zoom the image out with the #14 ratio. 

    _Note that if after zooming out, the image is still larger than the window size(1400,960), the displaying image would not bechanged. But, actually, the effect is applied correctly._
7. __Original image:__ return the image which displayed on #17 to the unprocessed form.
8. __Sigma of Gaussian:__ The parameter refers to the sigma in 2d gaussian function. 

    ![](https://imgur.com/0RHReXN.jpg)

    And, the Gaussian blur would use the gaussian function to make the desire filter.
9. __Kernerl size:__ The parameter refers to the size of Gaussian blur filter. If set this to 3, the program would adopt a 3*3 filter to the image.
10. __Apply:__ After setting the #3 and #4 parameters, push this button to apply the Gaussian blur effect on the image file.
11. __threshold:__ This parameter is used to filter out the lower gray scale value of magnitude of gradient, which may be seen as noise.
12. __magnitude or direction of gradient:__ This combo box can choose to apply magnitude or direction of gradient. The example results are as below.
![](https://imgur.com/859k9lL.jpg)
![](https://imgur.com/XDXDQn4.jpg)
13.  __Apply:__ After setting the #6 and #7, push this button to apply sobel operator on the image file.
14. __structure tensor window:__ This parameter refers to the window size w in the math equation. ![](https://imgur.com/iVa9Ass.jpg)

    If set this to 3, the program would adopt a 3*3 window size to compute the corner response of the image.
15. __window size of NMS:__ This window size is using for NMS operation. It means that in the assign window size, only a pixel of max corner response could be show up. If set this to 0 or 1, NMS would not be apply.
16. __Apply:__ After setting the #9 and #10, push this button to apply structure tensor on the image file.

## Usage

```bash
python3 main.py
```
__To ensure the correct of porgram the process should follow the order as below:__ 

rotation/scaling -> gaussian blur -> Sobel edge detect -> structure tensor
## Image

### Default Image Location

The default location is as same as the program file. However it is also able to load files in other directories through dialog window.

### Support Image Format

The application could load image files with `*.png, *.tif, *.bmp, *jpg` extension.

## Dependencies

* [Numpy](http://www.numpy.org/)

* [Opencv-Python](https://pypi.org/project/opencv-python/)

* [PyQt5](https://riverbankcomputing.com/software/pyqt/intro)

* [Scipy](https://pypi.org/project/scipy/)

## Reference
* [Gaussian smooth](http://www.librow.com/articles/article-9)

* [Sobel edge detection](https://medium.com/datadriveninvestor/understanding-edge-detection-sobel-operator-2aada303b900)

* [Compute structure tensor](https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html)

* [Non-maximal suppression](http://www.ipol.im/pub/art/2018/229/article_lr.pdf)