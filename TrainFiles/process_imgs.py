import csv
import copy
import cv2
import os

# Tento script prejde vsetky obrazky a rozdeli ich do potrebnych skupin
with open('list_attr_celeba.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    keys = []

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            for key in row.keys():
                keys.append(key)
            keys.remove("image_id")
            dict = dict((k, 0) for k in keys)
            for key in keys:
                if not os.path.exists("classes/" + key):
                    os.makedirs("classes/" + key)

        for key in keys:
            if row[key] == '1':
                tmp_img = cv2.imread("img_align_celeba/" + row["image_id"])
                clone_img = copy.copy(tmp_img)
                cv2.imwrite("classes/" + key + "/" + row["image_id"], clone_img)
                dict[key] += 1

        line_count += 1

    print(f'Processed {line_count} lines.')