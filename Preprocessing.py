import cv2

smallerImgWidth = 720/3
smallerImgHeight = 1280/3

# img is a 1280x720 image
def preprocessImage(img):
    smallerImg = cv2.resize(img, (smallerImgWidth, smallerImgHeight), interpolation=cv2.INTER_AREA)
    return smallerImg
