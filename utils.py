import cv2, os, random, time
import numpy as np
import torch

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

def matchingDebug(dir, path1, path2, K1, K2, matches, margin = 10):
    '''
    time_now = time.strftime("%Y%m%d-%H-%M", time.localtime())    
    if not os.path.exists("./debug/" + time_now + "/"):
        os.makedirs("./debug/" + time_now + "/")
    img1 = cv2.imread(os.path.join(dir, path1))
    img2 = cv2.imread(os.path.join(dir, path2))
    savePath = "./debug/" + time_now + "/" + path1 + "-" + path2 + ".png"
    '''
    img1 = cv2.imread(os.path.join(dir, path1))
    img2 = cv2.imread(os.path.join(dir, path2))
    savePath = "./debug/" + path1 + "-" + path2 + ".png"
    H1, W1, C1 = img1.shape
    H2, W2, C2 = img2.shape
    imageMatchShow = 255 * np.ones((H1, W1 + W2 + margin, 3), np.uint8)
    imageMatchShow[:H1, :W1, :] = img1
    imageMatchShow[:H2:, W1+margin:, :] = img2
    for i in range(len(matches)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        K1[matches[i][0]][0]
        cv2.line(imageMatchShow,
             (int(K1[matches[i][0]][0]), int(K1[matches[i][0]][1])),
             (int(K2[matches[i][1]][0]) + margin + W1, int(K2[matches[i][1]][1])),
             (r, g, b),
             thickness=int(H1/600),
             lineType=cv2.LINE_AA)
        cv2.circle(imageMatchShow,
             (int(K1[matches[i][0]][0]), int(K1[matches[i][0]][1])),
             int(H1/300),
             (r, g, b),
                   1)
        cv2.circle(imageMatchShow,
             (int(K2[matches[i][1]][0]) + margin + W1, int(K2[matches[i][1]][1])),
             int(H1/300),
             (r, g, b),
                   1)
    cv2.imwrite(savePath, imageMatchShow)
    print(savePath)