
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
from pathlib import Path
from database import COLMAPDatabase
import argparse
import numpy as np
import cv2
import matchers
from RPFeatDetectors import get_rpfeat_from_scenes_return

camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'FULL_OPENCV': 5,
                'SIMPLE_RADIAL_FISHEYE': 6,
                'RADIAL_FISHEYE': 7,
                'OPENCV_FISHEYE': 8,
                'FOV': 9,
                'THIN_PRISM_FISHEYE': 10}


def operate(cmd):
    print(cmd)
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    duration = end - start
    print("[%s] cost %f s" % (cmd, duration))


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_init_cameraparams(width, height, modelId):
    f = max(width, height) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    if modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([f, f, cx, cy])
    elif modelId == 2 or modelId == 6:
        return np.array([f, cx, cy, 0.0])
    elif modelId == 3 or modelId == 7:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId == 4 or modelId == 8:
        return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0])
    elif modelId == 9:
        return np.array([f, f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def init_cameras_database(db, images_path, cameratype, single_camera):
    print("init cameras database ......................................")
    images = []
    width = None
    height = None
    for name in sorted(os.listdir(images_path)):
        if 'jpg' in name or 'png' in name:  # 读取 jpg png 文件
            images.append(name)
            if width is None:
                img = cv2.imread(os.path.join(images_path, name))
                height, width = img.shape[:2]
    # 获取相机模型
    cameraModel = camModelDict[cameratype]
    params = get_init_cameraparams(width, height, cameraModel)
    # 单相机/多相机 模式
    if single_camera:
        db.add_camera(cameraModel, width, height, params, camera_id=0)
    for i, name in enumerate(images):
        if single_camera:
            db.add_image(name, 0, image_id=i)
            continue
        db.add_camera(cameraModel, width, height, params, camera_id=i)
        db.add_image(name, i, image_id=i)
    return images


def getFeatures(db, dir, images_name, matcherID=0, with_mask=False):
    images_path = os.path.join(dir, args.images_path)
    mask_path = os.path.join(args.dir, 'masks')
    match_list = open(match_list_path, 'w')
    num_images = len(images_name)
    for i in range(0, num_images):
        for j in range(i+1, num_images):
            match_list.write("%s %s\n" % (images_name[i], images_name[j]))
    match_list.close()
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")

    print("feature extraction...........................")
    features = get_rpfeat_from_scenes_return(images_path)

    for i, name in enumerate(images_name):
        keypoints = features[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
                                    np.ones((n_keypoints, 1)).astype(
                                        np.float32),
                                    np.zeros((n_keypoints, 1)).astype(np.float32)],
                                   axis=1)
        if with_mask:
            mask = cv2.imread(os.path.join(mask_path, Path(
                name).stem + '.png'), cv2.IMREAD_GRAYSCALE)
            index_invalid = []
            for ii in range(keypoints.shape[0]):
                if mask[round(keypoints[ii][0])][round(keypoints[ii][1])] < 128:
                    index_invalid.append(ii)
            keypoints = np.delete(keypoints, index_invalid, axis=0)
            features[name]['keypoints'] = keypoints
            features[name]['descriptors'] = np.delete(
                features[name]['descriptors'], index_invalid, axis=0)
        print(name, " : ", len(keypoints))
        db.add_keypoints(i, keypoints)

    print("match features by exhaustive match............................")
    start = time.perf_counter()
    if matcherID == 0:
        for i in range(0, num_images):
            for j in range(i+1, num_images):
                D1 = features[images_name[i]]['descriptors'] * 1.0
                D2 = features[images_name[j]]['descriptors'] * 1.0
                matches = matchers.mutual_nn_ratio_matcher(
                    D1, D2, 0.9).astype(np.uint32)
                db.add_matches(i, j, matches)
            print("\r{0}, Time:{1}min".format(
                ((i+1)/num_images * 2 - (i+1)/num_images * (i+1)/num_images),
                (time.perf_counter() - start)/60),
                end=""
            )
    if matcherID == 1:
        from adalam import AdalamFilter
        AdalamMatcher = AdalamFilter()
        AdalamMatcher.config['scale_rate_threshold'] = None
        AdalamMatcher.config['orientation_difference_threshold'] = None
        AdalamMatcher.config['min_inliers'] = 2
        for i in range(0, num_images):
            for j in range(i+1, num_images):
                K1 = features[images_name[i]]['keypoints']
                K2 = features[images_name[j]]['keypoints']
                K1 = K1[:, :2]
                K2 = K2[:, :2]
                D1 = features[images_name[i]]['descriptors'] * 1.0
                D2 = features[images_name[j]]['descriptors'] * 1.0
                shape1 = [features[images_name[i]]['image_size']
                          [1], features[images_name[i]]['image_size'][0]]
                shape2 = [features[images_name[j]]['image_size']
                          [1], features[images_name[j]]['image_size'][0]]
                matches = AdalamMatcher.match_and_filter(
                        K1, K2, D1, D2, im1shape=shape1, im2shape=shape2).data.cpu().numpy().astype(np.uint32)
                db.add_matches(i, j, matches)
            print("\r{0}, Time:{1}min".format(
                ((i+1)/num_images * 2 - (i+1)/num_images * (i+1)/num_images),
                (time.perf_counter() - start)/60),
                end=""
            )
    print("\n\n==============  Over  ==============\n\n")


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='Deep Colmap')
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--cameraModel", required=False,
                        type=str, default="SIMPLE_RADIAL")  # SIMPLE_RADIAL
    parser.add_argument("--images_path", required=False,
                        type=str, default="images")
    parser.add_argument("--single_camera", default=True)
    parser.add_argument("--match", type=int, default=0,
                        help='0-MutualRatio 1-Adalam')
    parser.add_argument("--with_mask", default=False)
    args = parser.parse_args()
    database_path = os.path.join(args.dir, "database.db")
    match_list_path = os.path.join(args.dir, "image_pairs_to_match.txt")
    images_path = os.path.join(args.dir, args.images_path)
    if os.path.exists(database_path):
        cmd = "rm %s" % database_path
        operate(cmd)
    # 新建数据库
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    # 初始化数据库
    images_name = init_cameras_database(
        db, images_path, args.cameraModel, args.single_camera)
    # 特征提取匹配
    getFeatures(db, args.dir, images_name,
                matcherID=args.match, with_mask=args.with_mask)
    db.commit()
    db.close()
