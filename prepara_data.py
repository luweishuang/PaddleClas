import os
import cv2

# index = 0
# src_dir = "/root/data/pfc/data/test"
# dst_dir = src_dir + "_new"
# os.makedirs(dst_dir, exist_ok=True)
# for sub in os.listdir(src_dir):
#     sub_path = os.path.join(src_dir, sub)
#     sub_path_dst = os.path.join(dst_dir, sub)
#     os.makedirs(sub_path_dst, exist_ok=True)
#     for cur_f in os.listdir(sub_path):
#         cur_img = os.path.join(sub_path, cur_f)
#         cur_img_dst = os.path.join(sub_path_dst, "%05d.jpg" % index)
#         index += 1
#         os.system("mv %s %s" % (cur_img, cur_img_dst))


src_dir = "/root/data/pfc/data/test"
src_dict = {"yesemp":"1", "noemp":"0"}
all_list = []
for sub, value in src_dict.items():
    sub_path = os.path.join(src_dir, sub)
    for cur_f in os.listdir(sub_path):
        cur_l = os.path.basename(src_dir) + "/" + sub + "/" + cur_f + " " + value
        all_list.append(cur_l)

src_txt = src_dir + "_list.txt"
with open(src_txt, "w") as fw:
    for cur_l in all_list:
        fw.write(cur_l + "\n")


# src_dir = "/data/ieemoo/judgeEmpty/data/test"
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
#         img_dst = cv2.resize(pic, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
#         cv2.imwrite(cur_img_dst, img_dst)