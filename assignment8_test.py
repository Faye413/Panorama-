import sys
import os
import numpy as np
import cv2

import assignment8

def test_getImageCorners():
    """ This script will robustly test the getImageCorners function.
    """
    matrix_1 = np.zeros((320, 320, 3))
    matrix_1_answer = np.array([[[0, 0]],
                                [[0, 320]],
                                [[320, 0]],
                                [[320, 320]]], dtype=np.float32)
    matrix_2 = np.zeros((10, 11, 3))
    matrix_2_answer = np.array([[[0, 0]],
                                [[0, 10]],
                                [[11, 0]],
                                [[11, 10]]], dtype=np.float32)
    matrix_3 = np.zeros((15, 17, 1))
    matrix_3_answer = np.array([[[0, 0]],
                                [[0, 15]],
                                [[17, 0]],
                                [[17, 15]]], dtype=np.float32)
    matrix_4 = np.zeros((17, 15, 1))
    matrix_4_answer = np.array([[[0, 0]],
                                [[0, 17]],
                                [[15, 0]],
                                [[15, 17]]], dtype=np.float32)
    matrices = [matrix_1, matrix_2, matrix_3, matrix_4]
    matrices_ans = [matrix_1_answer, matrix_2_answer,
                    matrix_3_answer, matrix_4_answer]
    print "Evaluating getImageCorners."

    for matrix_idx in range(len(matrices)):
        corners = assignment8.getImageCorners(matrices[matrix_idx])
        ans = matrices_ans[matrix_idx]

        # Test for type.
        if not type(corners) == type(ans):
            raise TypeError(
                ("Error - corners has type {}." + 
                 " Expected type is {}.").format(type(corners), type(ans)))
        # Test for shape.
        if not corners.shape == ans.shape:
            raise ValueError(
                ("Error - corners has shape {}." +
                 " Expected shape is {}.").format(corners.shape, ans.shape))

        # Test for type of values in matrix.
        if not type(corners[0][0][0]) == type(ans[0][0][0]):
            raise TypeError(
                ("Error - corners values have type {}." +
                 "Expected type is {}.").format(type(corners[0][0][0]),
                                                type(ans[0][0][0])))

        # Assert values are identical.
        supplied_corners = []
        for corner in corners:
            if corner.tolist() not in ans.tolist():
                raise ValueError(
                    ("Error - Supplied corner: {} not a correct corner." + \
                     "\nExpected corners are {}.").format(corner, ans))
            elif corner.tolist() in supplied_corners:
                raise ValueError(
                    "Error - Corner {} is repeated.".format(corner))
            else:
                supplied_corners.append(corner.tolist())
    print "getImageCorners tests passed."
    return True

def test_findMatchesBetweenImages():
    """ This script will perform a unit test on the matching function.
    """
    # Hard code output matches.
    image_1 = cv2.imread("images/source/panorama_1/1.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")

    print "Evaluating findMatchesBetweenImages."

    image_1_kp, image_2_kp, matches = \
        assignment8.findMatchesBetweenImages(image_1, image_2, 20)

    if not type(image_1_kp) == list:
        raise TypeError(
            "Error - image_1_kp has type {}. Expected type is {}.".format(
                type(image_1_kp), list))

    if len(image_1_kp) > 0 and \
        not type(image_1_kp[0]) == type(cv2.KeyPoint()):
        raise TypeError(("Error - The items in image_1_kp have type {}. " + \
                         "Expected type is {}.").format(type(image_1_kp[0]),
                                                        type(cv2.KeyPoint())))

    if not type(image_2_kp) == list:
        raise TypeError(
            "Error - image_2_kp has type {}. Expected type is {}.".format(
                type(image_2_kp), list))

    if len(image_2_kp) > 0 and \
        not type(image_2_kp[0]) == type(cv2.KeyPoint()):
        raise TypeError(("Error - The items in image_2_kp have type {}. " + \
                         "Expected type is {}.").format(type(image_2_kp[0]),
                                                        type(cv2.KeyPoint())))

    if not type(matches) == list:
        raise TypeError(
            "Error - matches has type {}. Expected type is {}. ".format(
                type(matches), list))

    if len(matches) > 0 and not type(matches[0]) == type(cv2.DMatch()):
        raise TypeError(("Error - The items in matches have type {}. " + \
                         "Expected type is {}.").format(type(matches[0]),
                                                        type(cv2.DMatch())))
    print "findMatchesBetweenImages testing passed."
    return True

def test_findHomography():
    # Hard code output matches.
    image_1 = cv2.imread("images/source/panorama_1/1.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")

    image_1_kp, image_2_kp, matches = assignment8.findMatchesBetweenImages(
        image_1, image_2, 20)

    homography = assignment8.findHomography(image_1_kp, image_2_kp, matches)

    ans = np.array([[1.43341688e+00, 2.97957049e-02, -1.10725647e+03],
                    [1.71655562e-01, 1.36708332e+00, -5.06635922e+02],
                    [1.06589023e-04, 6.16589688e-05, 1.00000000e+00]])

    print "Evaluating findHomography."
    # Test for type.
    if not type(homography) == type(ans):
        raise TypeError(
            ("Error - homography has type {}. " + 
             "Expected type is {}.").format(type(homography), type(ans)))
    # Test for shape.
    if not homography.shape == ans.shape:
        raise ValueError(
            ("Error - homography has shape {}." +
             " Expected shape is {}.").format(homography.shape, ans.shape))
    if not type(homography[0][0]) == type(ans[0][0]):
        raise TypeError(
            ("Error - The items in homography have type {}. " + 
             "Expected type is {}.").format(type(homography), type(ans)))

    # if not np.max(abs(np.subtract(homography, ans))) < 2.0:
    #     print "If your homography looks significantly different make sure " + \
    #           "you look at the warped image output. That is the only way " + \
    #           "of knowing if it is correct. Images should be aligned well. " + \
    #           "We expect ORB & SIFT to output equivalent homographies, but " + \
    #           "if you run into this error please take a look at your output."
    #     raise ValueError(
    #         ("Error - your output seems to be significantly different, " +
    #          "please verify this yourself. \n Given output {}. \n" +
    #          "Expected output {}. \n").format(homography, ans))
    print "findHomography testing passed."
    return True

def test_blendImagePair():
    warped_image = cv2.imread("images/testing/warped_image.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")
    point = (1107.26, 506.64)

    blended = assignment8.blendImagePair(warped_image, image_2, point)

    type_answer = np.copy(warped_image)
    print type_answer.shape
    type_answer[point[1]:point[1] + image_2.shape[0],
                point[0]:point[0] + image_2.shape[1]] = image_2

    print "Evaluating blendImagePair"
    # Test for type.
    if not type(blended) == type(type_answer):
        raise TypeError(
            ("Error - blended_image has type {}. " +
             "Expected type is {}.").format(type(blended), type(type_answer)))

    # Test for shape.
    if not blended.shape == type_answer.shape:
        raise ValueError(
            ("Error - blended_image has shape {}. " +
             "Expected shape is {}.").format(blended.shape, type_answer.shape))

    # Check if output is equivalent.
    if np.array_equal(blended, type_answer):
        print "WARNING: Blended image function has not been changed. You " + \
              "need to add your own functionality or you will not get " + \
              "credit for its implementation."

    print "blendImagePair testing passed."
    return True

def test_warpImagePair():
    image_1 = cv2.imread("images/source/panorama_1/1.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")
    image_1_kp, image_2_kp, matches = assignment8.findMatchesBetweenImages(
        image_1, image_2, 20)
    homography = assignment8.findHomography(image_1_kp, image_2_kp, matches)
    warped_image = assignment8.warpImagePair(image_1, image_2, homography)

    # Read in answer that has the correct type / shape.
    type_answer = cv2.imread("images/testing/warped_image_1_2.jpg")

    print "Evaluating warpImagePair."
    # Test for type.
    if not type(warped_image) == type(type_answer):
        raise TypeError(
            ("Error - warped_image has type {}. " + 
             "Expected type is {}.").format(type(warped_image),
                                            type(type_answer)))
    # Test for shape.
    if abs(np.sum(np.subtract(warped_image.shape, type_answer.shape))) > 200:
        print ("WARNING - warped_image has shape {}. " +
               "Expected shape is around {}.").format(warped_image.shape,
                                                      type_answer.shape)
    print "warpImagePair testing passed."

    return True

if __name__ == "__main__":
    print "Performing unit test."
    if not test_getImageCorners():
        print "getImageCorners function failed. Halting testing."
        sys.exit()
    if not test_findMatchesBetweenImages():
        print "findMatchesBetweenImages function failed. Halting testing."
        sys.exit()
    if not test_findHomography():
        print "findHomography function failed. Halting testing."
        sys.exit()
    if not test_blendImagePair():
        print "blendImagePair function failed. Halting testing."
        sys.exit()
    if not test_warpImagePair():
        print "warpImagePair function failed. Halting testing."
        sys.exit()
    print "Unit test passed."

  
    sourcefolder = os.path.abspath(os.path.join(os.curdir, "images", "source"))
    outfolder = os.path.abspath(os.path.join(os.curdir, "images", "output"))

    print "Image source folder: {}".format(sourcefolder)
    print "Image output folder: {}".format(outfolder)

    print "Searching for folders with images in {}.".format(sourcefolder)

    # Extensions recognized by opencv
    exts = [".bmp", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".jpeg", ".jpg", 
            ".jpe", ".jp2", ".tiff", ".tif", ".png"]

    # For every image in the source directory
    for dirname, dirnames, filenames in os.walk(sourcefolder):
        setname = os.path.split(dirname)[1]

        panorama_inputs = []
        panorama_filepaths = []

        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext.lower() in exts:
                panorama_filepaths.append(os.path.join(dirname, filename))
        panorama_filepaths.sort()

        for pan_fp in panorama_filepaths:
            panorama_inputs.append(cv2.imread(pan_fp))

        if len(panorama_inputs) > 1:
            print ("Found {} images in folder {}. " + \
                   "Processing them.").format(len(panorama_inputs), dirname)
        else:
            continue

        print "Computing matches."
        cur_img = panorama_inputs[0]
        for new_img in panorama_inputs[1:]:
            image_1_kp, image_2_kp, matches = \
                assignment8.findMatchesBetweenImages(cur_img, new_img, 20)
            print "Computing homography."
            homography = assignment8.findHomography(image_1_kp, image_2_kp,
                                                    matches)
            print "Warping the image pair."
            cur_img = assignment8.warpImagePair(cur_img, new_img, homography)
        
        print "Writing output image to {}".format(outfolder)
        cv2.imwrite(os.path.join(outfolder, setname) + ".jpg", cur_img)
