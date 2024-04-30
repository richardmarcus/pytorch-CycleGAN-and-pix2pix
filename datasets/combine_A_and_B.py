import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, -1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, -1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR

    #modified this to generate the paired images from the panorama images but not required for ex1

    #only take inner 1/32 width of image so left is 15/32 and right is 17/32
    im_A = im_A[:, int(im_A.shape[1]*15/32):int(im_A.shape[1]*17/32)]
    im_B = im_B[:, int(im_B.shape[1]*15/32):int(im_B.shape[1]*17/32)]
    

    im_AB = np.concatenate([im_A, im_B], 1)
    #print(im_AB.shape)
    cv2.imwrite(path_AB, im_AB)


    #insert ambient_ and reflec_ after the / before the last / it is not known if the path is /train or /test
    path_AB_ambient = path_AB.replace("/train/", "/ambient_train/")
    path_AB_ambient = path_AB_ambient.replace("/test/", "/ambient_test/")
    path_AB_reflec = path_AB.replace("/train/", "/reflec_train/")
    path_AB_reflec = path_AB_reflec.replace("/test/", "/reflec_test/")

    
    cv2.imwrite(path_AB_ambient, im_A)
    print(path_AB_ambient)

    cv2.imwrite(path_AB_reflec, im_B)
    print(path_AB_reflec)
  

    print(path_AB)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--fold_AB_test', dest='fold_AB_test', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

#splits = os.listdir(args.fold_A)
splits = [""]

#makedir ambient_train, ambient_test, reflec_train, reflec_test
#ambient_train path: replace /train with /ambient_train and so on

ambient_train = args.fold_AB.replace("/train", "/ambient_train")
ambient_test = args.fold_AB_test.replace("/test", "/ambient_test")
reflec_train = args.fold_AB.replace("/train", "/reflec_train")
reflec_test = args.fold_AB_test.replace("/test", "/reflec_test")

if not os.path.isdir(ambient_train):
    os.makedirs(ambient_train)
if not os.path.isdir(ambient_test):
    os.makedirs(ambient_test)
if not os.path.isdir(reflec_train):
    os.makedirs(reflec_train)
if not os.path.isdir(reflec_test):
    os.makedirs(reflec_test)

#delete all files in fold_ab and fold_ab_test if they exist
for sp in splits:
    img_fold_AB = os.path.join(args.fold_AB, sp)
    img_fold_AB_test = os.path.join(args.fold_AB_test, sp)
    if os.path.isdir(img_fold_AB):
        for file in os.listdir(img_fold_AB):
            os.remove(os.path.join(img_fold_AB, file))
    if os.path.isdir(img_fold_AB_test):
        for file in os.listdir(img_fold_AB_test):
            os.remove(os.path.join(img_fold_AB_test, file))   




if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_list = os.listdir(img_fold_A)
    #print(img_fold_A)
    #shuffle
    np.random.seed(0)
    np.random.shuffle(img_list)

    #sort
    #img_list.sort()

    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        if n>1000:
            img_fold_AB = os.path.join(args.fold_AB_test, sp)
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = os.path.join(img_fold_AB, name_AB)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_AB))
            else:
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)
if not args.no_multiprocessing:
    pool.close()
    pool.join()
