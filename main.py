import cv2
import numpy as np
import matplotlib.image as mpimg
import glob
import re

def slopeAveraging(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # print(slope)
        if slope < 0:
            # if(slope>-0.9 and slope > -0.6)
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))


    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = makeCoordinates(image, left_fit_average)
    right_line = makeCoordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def makeCoordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def sobelEdgeDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5, 5), 0)
    sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    #edges = cv2.convertScaleAbs(sobelxy,alpha=1, beta=0)
    #edges =  cv2.threshold(edges,50,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #edges = cv2.dilate(edges,(3,3))
    
    return sobelxy

def lineIntersection(lines):
    xdiff = (lines[0][0] - lines[0][2], lines[1][0] - lines[1][2])
    ydiff = (lines[0][1] - lines[0][3], lines[1][1] - lines[1][3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return 0, 0

    d = (det(*((lines[0][0],lines[0][1]),(lines[0][2],lines[0][3]))), det(*((lines[1][0],lines[1][1]),(lines[1][2],lines[1][3]))))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)

    return x, y

def generateLines(image, lines):
    endx,endy = lineIntersection(lines)

    twoLanes = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(twoLanes, (x1, y1), (endx, endy), (255, 0, 0), 5)

    cv2.circle(twoLanes, (endx,endy), 10, (255, 255, 0), 2)
    return twoLanes

def processArea(image):
    polygons = np.array([[(0, image.shape[0]),(0, image.shape[0]-100), (290, 320), (340, 320), (image.shape[1]-150, image.shape[0])]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(img):
    pixels = np.array(img)
    height, width = pixels.shape
    diag_len = int(np.sqrt(height**2 + width**2))
    hough_space = np.zeros((2*diag_len, 180), dtype = np.uint64)
    thetas = np.deg2rad(np.arange(-90,90))
    for y in range (height):
        for x in range (width):
            if pixels[y, x]>128:
                for theta_index, theta in enumerate(thetas):
                    rho = int(x*np.cos(theta) + y * np.sin(theta)) + diag_len
                    hough_space[rho, theta_index] += 1

    return hough_space, thetas, diag_len

def find_lines (hough_space, thetas, diag_len, threshold = 100):
    peaks = np.where(hough_space >threshold)
    lines = []
    for i in range(len(peaks[0])):
        rho = peaks[0][i] - diag_len
        theta = thetas[peaks[1][i]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines.append(np.array([(x1, y1), (x2, y2)]))  # Convert to NumPy array
    return lines

def detectLanes(img, threshold = 100):
    sobel = sobelEdgeDetection(img)
    cropped_Image = processArea(sobel)
    edges = cv2.convertScaleAbs(cropped_Image,alpha=1, beta=0)
    thresh,edges =  cv2.threshold(edges,50,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.dilate(edges,(3,3))
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, np.array([]), minLineLength=60, maxLineGap=10)
    #hough_space, thetas, diag_len = hough_transform(cropped_Image)
    #lines = find_lines(hough_space, thetas, diag_len, threshold=100)
    
    filteredLines = slopeAveraging(img, lines)
    twoLanes = generateLines(img, filteredLines)
    finalImg = cv2.addWeighted(img, 0.8, twoLanes, 1, 1)
    return finalImg

def videoOut():
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    images = []
    filenames = []
    #size = None
    for filename in glob.glob('Output3/*.jpg'):
        filenames.append(filename)
        img = cv2.imread(filename)\

    filenames.sort(key=natural_keys)
          
    for file in filenames:  
        img = cv2.imread(file)
        height, width, layers = img.shape
        size = (width,height)
        images.append(img)

    out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    
    for i in range(len(images)):
        out.write(images[i])
    out.release()

paths = glob.glob('TestVideo_1/*.bmp')

for i,image_path in enumerate(paths):
    print(image_path)
    image = mpimg.imread(image_path)
    img = np.copy(image)
    result = detectLanes(img)
    mpimg.imsave('Output2/'+image_path[12:-4]+'_out.jpg', result)

videoOut()
