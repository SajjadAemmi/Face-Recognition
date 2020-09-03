from face3d import mesh_numpy as mesh
# from utils.cython import

import cv2

import math
import statistics

import numpy as np

from utils_pose.estimate_pose import estimate_pose
from utils_pose.rotate_vertices import frontalize
from utils_pose.render_app import get_visibility, get_uv_mask, get_depth_image
from utils_pose.write import write_obj_with_colors, write_obj_with_texture



def transform_test(vertices, triangles, colors, obj, camera, h=256, w=256):
    '''
    Args:
        obj: dict contains obj transform paras
        camera: dict contains camera paras
    '''
    R = mesh.transform.angle2matrix(obj['angles'])
    transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])

    if camera['proj_type'] == 'orthographic':
        projected_vertices = transformed_vertices
        image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    else:

        ## world space to camera space. (Look at camera.)
        camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
        ## camera space to image space. (Projection) if orth project, omit
        projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near=camera['near'],
                                                                far=camera['far'])
        ## to image coords(position in image)
        image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

    print(image_vertices.shape)


    rendering = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
    print(rendering.shape)

    rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return rendering,transformed_vertices


def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d


def compute_similarity_transform(points_static, points_to_transform):
    #http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(3,1)
    t_final = t1 -t0

    p0c = p0+t0
    p1c = p1+t1

    covariance_matrix = p0c.dot(p1c.T)
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))

    s = (rms_d0/rms_d1)
    P = np.c_[s*np.eye(3).dot(R), t_final]
    return P


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def make_3D_Model(imagename,facedetector,landsdetector):
    #############################################################
    # Get image and recinstruct 3D model
    ### load image
    image_filename = imagename
    print(image_filename)
    image_face = cv2.imread(image_filename)
    # image_face = cv2.resize(image_face, (width, height))


    ### Build 3D model

    img, infs = facedetector.detectFacesFromImage(input_image=image_face, box_mark=False)

    if img != []:
        cv2.imshow("Input face", image_face)

        dets = []
        for inf in infs:
            print(inf)
            dets.append(inf["detection_details"])
            image_face = np.asarray(image_face)
            cv2.rectangle(image_face, (inf["detection_details"][0], inf["detection_details"][1]),
                          (inf["detection_details"][2], inf["detection_details"][3]), color=(0, 255, 255), thickness=2)
            cv2.imshow("Input face", image_face)

        # for i in range(0,20):
        #  out.write(cv2.resize(image_face, (width, height)))

        print('the number of faces: {:0>3d}'.format((len(infs))))

        # loop over the face detections
        # for rect in dets:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy

        img3d = landsdetector.restructure3DFaceFromImage(img, dets=dets, pose=True, output_path="./Results")

    cv2.waitKey(5)

