import numpy as np
import cv2

def blue_only(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    # Threshold of blue in HSV space
    #lower_blue = np.array([60, 35, 140])
    #upper_blue = np.array([180, 255, 255])
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([150, 255, 255])
 
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask = mask)
    return result

def _blue_only(img):
    rows, cols, chs = img.shape
    for row in range(rows):
        for col in range(cols):
            b,g,r = img[row][col]
            if b > 20 and b * 0.65 > g and b * 0.65 > r:
                pass 
            else:
                gray = b/3 + g/3 + r/3
                img[row][col] = gray,gray,gray
    return img
def blue_yellow(img):
    rows, cols, chs = img.shape
    for row in range(rows):
        for col in range(cols):
            b,g,r = img[row][col]
            if b > 20 and b * 0.65 > g and b * 0.65 > r:
                pass
            elif (r * 0.2 + g * 0.2) > b and abs(int(r) - int(g)) < 30:
                pass
            else:
                gray = b/3 + g/3 + r/3
                img[row][col] = gray,gray,gray
    return img

if __name__ == '__main__':
    img = cv2.imread('nctu_flag.jpg')
    b_img = img.copy()
    by_img = img.copy()

    rows,cols,chs = img.shape
    b_img = blue_only(b_img,rows,cols,chs)
    by_img = blue_yellow(by_img,rows,cols,chs)


    #cv2.imshow("image",by_img)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite('blue-only.jpg',b_img)
    cv2.imwrite('blue-yellow.jpg',by_img);
