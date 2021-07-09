import os
import cv2
import numpy as np
import random


# index = 0
# src_dir =  "/data/ieemoo/judgeEmpty/new_data_5cls/train"
# dst_dir = src_dir + "_new"
# os.makedirs(dst_dir, exist_ok=True)
# for sub in os.listdir(src_dir):
#     sub_path = os.path.join(src_dir, sub)
#     sub_path_dst = os.path.join(dst_dir, sub)
#     os.makedirs(sub_path_dst, exist_ok=True)
#     for cur_f in os.listdir(sub_path):
#         cur_img = os.path.join(sub_path, cur_f)
#         cur_img_dst = os.path.join(sub_path_dst, "a%05d.jpg" % index)
#         index += 1
#         os.system("mv %s %s" % (cur_img, cur_img_dst))


# src_dir = "/data/ieemoo/judgeEmpty/new_data_5cls/train"
# src_dict = {"yesemp":"1", "noemp":"0", "hard": "2", "fly": "3", "stack": "4"}
# all_dict = {"yesemp":[], "noemp":[], "hard": [], "fly": [], "stack": []}
# for sub, value in src_dict.items():
#     sub_path = os.path.join(src_dir, sub)
#     for cur_f in os.listdir(sub_path):
#         cur_l = os.path.basename(src_dir) + "/" + sub + "/" + cur_f + " " + value
#         all_dict[sub].append(cur_l)
#
# yesnum = len(all_dict["yesemp"])
# nonum = len(all_dict["noemp"])
# hardnum = len(all_dict["hard"])
# flynum = len(all_dict["fly"])
# stacknum = len(all_dict["stack"])
# thnum = min(yesnum, nonum, hardnum, flynum, stacknum)
# src_txt = src_dir + "_list.txt"
# with open(src_txt, "w") as fw:
#     for feat_path in random.sample(all_dict["yesemp"], thnum):
#         fw.write(feat_path + "\n")
#
#     for feat_path in random.sample(all_dict["noemp"], thnum):
#         fw.write(feat_path + "\n")
#
#     for feat_path in random.sample(all_dict["hard"], thnum):
#         fw.write(feat_path + "\n")
#
#     for feat_path in random.sample(all_dict["fly"], thnum):
#         fw.write(feat_path + "\n")
#
#     for feat_path in random.sample(all_dict["stack"], thnum):
#         fw.write(feat_path + "\n")


# src_dir = "/data/ieemoo/judgeEmpty/new_data_5cls/train"
# dst_dir = src_dir + "_new"
# os.makedirs(dst_dir, exist_ok=True)
# for sub in os.listdir(src_dir):
#     sub_path = os.path.join(src_dir, sub)
#     sub_path_dst = os.path.join(dst_dir, sub)
#     os.makedirs(sub_path_dst, exist_ok=True)
#     for cur_f in os.listdir(sub_path):
#         cur_img = os.path.join(sub_path, cur_f)
#         cur_img_dst = os.path.join(sub_path_dst, cur_f)
#         pic = cv2.imread(cur_img)
#         height, width, channels = pic.shape
#         min_l = min(height, width)
#         if min_l <= 256:
#             # print(cur_img, height, width)
#             os.system("mv %s %s" % (cur_img, cur_img_dst))
#         else:
#             ratio_l = 256.0 / min_l
#             img_dst = cv2.resize(pic, (0, 0), fx=ratio_l, fy=ratio_l)
#             cv2.imwrite(cur_img_dst, img_dst)


# image = cv2.imread("/data/ieemoo/judgeEmpty/data/tt/Webcam/0a05d.jpg")
# cv2.imshow("image", image)
#
# image1 = cv2.flip(image, -1) #原图顺时针旋转180度
# cv2.imshow("0", image1)
#
# imageT=cv2.transpose(image)   #转置图像
# cv2.imshow("1", imageT)
#
# image2 = cv2.flip(image, 0)  #等于原图顺时针旋转270度
# cv2.imshow("22", image2)
#
# image3 = cv2.flip(image, 1)  #等于原图顺时针旋转90度
# cv2.imshow("33", image3)
#
# image4 = cv2.flip(imageT, 0)  #等于原图顺时针旋转270度
# cv2.imshow("4", image4)
#
# image5 = cv2.flip(imageT, 1)  #等于原图顺时针旋转90度
# cv2.imshow("5", image5)
# cv2.waitKey(600000)


# src_dir = "/data/ieemoo/judgeEmpty/data/tt"
# dst_dir = src_dir + "_new"
# os.makedirs(dst_dir, exist_ok=True)
# for sub in os.listdir(src_dir):
#     sub_path = os.path.join(src_dir, sub)
#     sub_path_dst = os.path.join(dst_dir, sub)
#     os.makedirs(sub_path_dst, exist_ok=True)
#     for cur_f in os.listdir(sub_path):
#         cur_img = os.path.join(sub_path, cur_f)
#
#         image = cv2.imread(cur_img)
#         cv2.imwrite(os.path.join(sub_path_dst, cur_f), image)
#
#         image1 = cv2.flip(image, -1)
#         cv2.imwrite(os.path.join(sub_path_dst, cur_f.replace(".jpg", "_1.jpg")), image1)
#
#         imageT = cv2.transpose(image)
#         cv2.imwrite(os.path.join(sub_path_dst, cur_f.replace(".jpg", "_t.jpg")), imageT)
#
#         image2 = cv2.flip(image, 0)
#         cv2.imwrite(os.path.join(sub_path_dst, cur_f.replace(".jpg", "_2.jpg")), image2)
#
#         image3 = cv2.flip(image, 1)
#         cv2.imwrite(os.path.join(sub_path_dst, cur_f.replace(".jpg", "_3.jpg")), image3)
#
#         image4 = cv2.flip(imageT, 0)
#         cv2.imwrite(os.path.join(sub_path_dst, cur_f.replace(".jpg", "_4.jpg")), image4)
#
#         image5 = cv2.flip(imageT, 1)
#         cv2.imwrite(os.path.join(sub_path_dst, cur_f.replace(".jpg", "_5.jpg")), image5)


src_txt = "/data/ieemoo/judgeEmpty/new_data_5cls/train_list_equal.txt"
dst_txt = src_txt.replace(".txt", "_new.txt")
with open(dst_txt, "w") as fw:
    with open(src_txt, "r") as fr:
        for c_l in fr:
            if "hard" in c_l or "fly" in c_l or "stack" in c_l:
                new_l = c_l.split(" ")[0].replace("hard", "noemp").replace("fly", "noemp").replace("stack", "noemp") + " " + "0"
                fw.write(new_l + "\n")
            else:
                fw.write(c_l)
