import numpy as np
import cv2

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def separateData(x, labels):
    glaucomaData = [i for i, l in enumerate(np.argmax(labels, 1) ) if l == 1 ]
    healthyData = [i for i, l in enumerate(np.argmax(labels, 1)) if l == 0 ]

    return glaucomaData, healthyData

def balancedDataAugmentation(x, labels, vessels, angles=[90, 180, 270]):
    from sklearn.model_selection import train_test_split

    data = []
    [data.append(img) for img in x]

    lab = []
    [lab.append(l) for l in labels]


    g, h = separateData(x, labels)

    amount = np.bincount(np.argmax(labels, 1))

    quant = (amount[0]-amount[1])/3.

    ves = np.zeros( (len(vessels) + int(quant)*len(angles), 112, 112, 1), 'float32')
    for i in range(len(vessels)):
        ves[i, :, :, :] = vessels [i, :, :, :]


    ind_ves = len(vessels)
    for i in range(int(quant) ):
        img = data[g[i]].copy()
        ves_img = ves[g[i]].copy()

        for angle in angles:
            newimg = rotate_image(img, angle)
            data.append(newimg)
            lab.append(labels[g[i]])

            newves = rotate_image(ves_img, angle)
            ves[ind_ves, :, :, :] = np.expand_dims( newves[:, :], 2 )
            ind_ves += 1

    return np.asarray(data).astype('float32'), np.asarray(lab).astype('float32'), ves