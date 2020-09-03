import sys
import numpy as np
import cv2
from imutils import face_utils
import datetime
import imutils
import time
import dlib

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList();

    delaunayTri = []

    pt = []

    count= 0

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            count = count + 1
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []


    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect




def face_swap3(img_ref, detector, predictor):

    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects1 = detector(gray1, 0)

    if (len(rects1) < 2): #at least 2 faces in image need to be found
        return None

    if is_out_of_image(rects1, gray1.shape[1], gray1.shape[0]):
        return None

    img1Warped = np.copy(img_ref);

    shape1 = predictor(gray1, rects1[0])
    points1 = face_utils.shape_to_np(shape1) #type is a array of arrays (list of lists)

    if is_out_of_image_points(points1, gray1.shape[1], gray1.shape[0]): #check if points are inside the image
        return None

    #need to convert to a list of tuples
    points1 = list(map(tuple, points1))

    shape2 = predictor(gray1, rects1[1])
    points2 = face_utils.shape_to_np(shape2)

    if is_out_of_image_points(points2, gray1.shape[1], gray1.shape[0]): #check if points are inside the image
        return None

    points2 = list(map(tuple, points2))

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[ int(hullIndex[i]) ])
        hull2.append(points2[ int(hullIndex[i]) ])


    # Find delanauy traingulation for convex hull points
    sizeImg2 = img_ref.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        return None

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img_ref, img1Warped, t1, t2)


    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img_ref.shape, dtype = img_ref.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))


    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img_ref, mask, center, cv2.NORMAL_CLONE)


    img1Warped = np.copy(img_ref);
    dt = calculateDelaunayTriangles(rect, hull1)

    if len(dt) == 0:
        return None

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull2[dt[i][j]])
            t2.append(hull1[dt[i][j]])

        warpTriangle(img_ref, img1Warped, t1, t2)


    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img_ref.shape, dtype = img_ref.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull1]))

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), output, mask, center, cv2.NORMAL_CLONE)

    return output


#put face in img_ref into face of img_mount_face
def face_swap(img_ref, img_mount_face, detector, predictor):

    gray2 = cv2.cvtColor(img_mount_face, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects2 = detector(gray2, 0)


    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects1 = detector(gray1, 0)

    rects2 = rects1

    w,h,_ = img_mount_face.shape
    rects2 = dlib.rectangle(0, 0, w, h)

    print(rects1[0])
    print(rects2)

    if (len(rects1) == 0): #if not found faces in images return error
        return None

    img1Warped = np.copy(img_mount_face);

    shape1 = predictor(gray1, rects1[0])
    points1 = face_utils.shape_to_np(shape1) #type is a array of arrays (list of lists)
    #need to convert to a list of tuples
    points1 = list(map(tuple, points1))
    shape2 = predictor(gray2, rects2)
    points2 = face_utils.shape_to_np(shape2)
    points2 = list(map(tuple, points2))

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[ int(hullIndex[i]) ])
        hull2.append(points2[ int(hullIndex[i]) ])


    # Find delanauy traingulation for convex hull points
    sizeImg2 = img_mount_face.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        return None

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img_ref, img1Warped, t1, t2)


    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img_mount_face.shape, dtype = img_mount_face.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))


    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img_mount_face, mask, center, cv2.NORMAL_CLONE)

    return output

#swaps faces in img_ref and img_mount_face (two separate files)
def face_swap2(img_ref, img_mount_face, detector, predictor):

    gray2 = cv2.cvtColor(img_mount_face, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects2 = detector(gray2, 0)


    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects1 = detector(gray1, 0)
    print(len(rects2))
    if (len(rects2) == 0 or len(rects1) == 0): #if not found faces in images return error
        return None, None

    img1Warped = np.copy(img_mount_face);
    img2Warped = np.copy(img_ref);

    shape1 = predictor(gray1, rects1[0])
    points1 = face_utils.shape_to_np(shape1) #type is a array of arrays (list of lists)
    #need to convert to a list of tuples
    points1 = list(map(tuple, points1))
    shape2 = predictor(gray2, rects2[0])
    points2 = face_utils.shape_to_np(shape2)
    points2 = list(map(tuple, points2))

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[ int(hullIndex[i]) ])
        hull2.append(points2[ int(hullIndex[i]) ])


    # Find delanauy traingulation for convex hull points
    sizeImg2 = img_mount_face.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        return None, None

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img_ref, img1Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img_mount_face.shape, dtype = img_mount_face.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))


    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img_mount_face, mask, center, cv2.NORMAL_CLONE)


    # Find delanauy traingulation for convex hull points
    sizeImg1 = img_ref.shape
    rect = (0, 0, sizeImg1[1], sizeImg1[0])
    dt = calculateDelaunayTriangles(rect, hull1)

    if len(dt) == 0:
        return None, None

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull2[dt[i][j]])
            t2.append(hull1[dt[i][j]])

        warpTriangle(img_mount_face, img2Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull1)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img_ref.shape, dtype = img_ref.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull1]))

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    # Clone seamlessly.
    output2 = cv2.seamlessClone(np.uint8(img2Warped), img_ref, mask, center, cv2.NORMAL_CLONE)

    return output, output2



def is_out_of_image(rects, imgW, imgH):
    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        if x < 0 or y <0 or (y+h) >= imgH or (x+w) >= imgW:
            return True
    return False

def is_out_of_image_points(points, imgW, imgH):
    for x,y in points:
        if x < 0 or y < 0 or y >= imgH or x >= imgW:
            return True
    return False


#put face in img_ref into face of img_mount_face
def face_swap_cropedimage(img_ref, face_ref_rect, img_mount_face, detector, predictor):

    gray2 = cv2.cvtColor(img_mount_face, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects2 = detector(gray2, 0)

    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    if (len(rects2) == 0): #if not found faces in images return error
        return None

    if is_out_of_image(rects2, gray2.shape[1], gray2.shape[0]):
        return None

    img1Warped = np.copy(img_mount_face);

    shape1 = predictor(gray1, face_ref_rect) #rects1 vienen de entrada afuera
    points1 = face_utils.shape_to_np(shape1) #type is a array of arrays (list of lists)

    #need to convert to a list of tuples
    points1 = list(map(tuple, points1))

    shape2 = predictor(gray2, rects2[0])
    points2 = face_utils.shape_to_np(shape2)
    if is_out_of_image_points(points2, gray2.shape[1], gray2.shape[0]): #check if points are inside the image
        return None
    points2 = list(map(tuple, points2))

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[ int(hullIndex[i]) ])
        hull2.append(points2[ int(hullIndex[i]) ])


    # Find delanauy traingulation for convex hull points
    sizeImg2 = img_mount_face.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        return None

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img_ref, img1Warped, t1, t2)


    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img_mount_face.shape, dtype = img_mount_face.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))


    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img_mount_face, mask, center, cv2.NORMAL_CLONE)

    return output
