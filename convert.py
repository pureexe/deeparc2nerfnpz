import numpy as np 
import matplotlib.pyplot as plt
import cv2, os
from read_write_model import read_model
from scipy.spatial.transform import Rotation

SQUARE_SIZE = 100
MODEL_DIR = 'model/'
IMAGE_DIR = 'images/'
OUTPUT_NPZ =  'npz/greentea.npz'

# @see https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# @see https://gist.github.com/jdhao/f8422980355301ba30b6774f610484f2
def pad_image(image, size_square):
    delta_h = size_square - image.shape[0]
    delta_w = size_square - image.shape[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_image

def image_square(image,size):
    if image.shape[0] > image.shape[1]:
        new_image = image_resize(image,height = size)
    else:
        new_image = image_resize(image,width = size)
    return pad_image(new_image, size)

def main():
    cameras, images, points3D = read_model(MODEL_DIR, '.bin')
    focal = 0.0
    width = 0.0
    height = 0.0
    for cam_id in cameras:
        width += cameras[cam_id][2]
        height += cameras[cam_id][3]
        focal += cameras[cam_id][4][0]
    width /= len(cameras)
    height/= len(cameras)
    focal /= len(cameras)
    """
    if height > width:
        focal *= SQUARE_SIZE / height
    else:
        focal *=SQUARE_SIZE / width
    """
    print("FOCAL = ", focal)
    exit()
    images_pixels = np.zeros((len(images),SQUARE_SIZE,SQUARE_SIZE,3),dtype=np.float32)
    poses = np.zeros((len(images),4,4),dtype=np.float32)
    for image_id in [102]:
        img_path = os.path.join(IMAGE_DIR,images[image_id][4])
        img = plt.imread(img_path)
        img = image_square(img,100)
        img = img.astype(np.float32)
        img /= 255.0
        images_pixels[image_id-1,:,:] = img
        qvec = images[image_id][1]
        tvec = images[image_id][2]
        rot = Rotation.from_quat([qvec[1],qvec[2],qvec[3],qvec[0]])
        poses[image_id - 1, :3, :3] = rot.as_matrix()
        poses[image_id - 1, :3, 3] = tvec
        poses[image_id - 1, 3, 3] = 1.0
        plt.imshow(img)
        plt.show()
    #np.savez(OUTPUT_NPZ, focal=focal, poses=poses, images=images_pixels)
    
    """
    image = plt.imread('images/cam000/cam000_00000.jpg')
    image = image_square(image,100)
    plt.imshow(image)
    plt.show()
    """

if __name__ == "__main__":
    main()