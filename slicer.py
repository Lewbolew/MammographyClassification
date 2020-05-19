import numpy as np
import cv2
import pandas as pd
import time
import os
from tqdm import tqdm


class Slicer:
    MIN_PIX_AMOUNT = 0

    def __init__(self, data, mask_folders: list, original_folder: str, patches_dir: str = "patches_data",
                 patch_size: tuple = (224, 224)):
        self.df = pd.read_csv(data)
        self.mask_folders = mask_folders
        self.original_folder = original_folder
        self.patches_dir = patches_dir
        self.patch_size = patch_size
        self.new_df_dict = {'Patch_path': [], 'Patch_label': []}

    def create_mask_img_pairs(self, df, mask_name):
        mask_img_pairs = []
        for index, row in df.iterrows():
            if pd.isna(row[mask_name]):
                continue
            mask_img_pairs.append({'img': row['IMG_PATH'], "mask": row[mask_name], "id": row["File Name"]})
        return mask_img_pairs

    def normal_images(self):
        normal_images_path = []
        for index, row in self.df.iterrows():
            if pd.isna(row['Calc path']) and pd.isna(row['Mask path']):
                normal_images_path.append({'img': row['IMG_PATH']})
        return normal_images_path

    def count_black_and_white(self, img):
        return np.sum(img == 255), np.sum(img == 0)

    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]

    def patches_from_mask(self):
        if not os.path.exists(self.patches_dir):
            os.mkdir(self.patches_dir)
        for item in self.mask_folders:
            label = item[0]
            mask_folder = item[1]
            mask_img_pairs = self.create_mask_img_pairs(self.df, item[2])
            patches_path = os.path.join(self.patches_dir, mask_folder.split('/')[-1])
            os.mkdir(patches_path)
            winW, winH = self.patch_size[0], self.patch_size[1]
            for pair in tqdm(mask_img_pairs):
                counter = 0
                mask_img = cv2.imread(os.path.join(mask_folder, pair['mask']))
                for (x, y, window) in self.sliding_window(mask_img, stepSize=50, windowSize=(winW, winH)):
                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue
                    white_p, black_p = self.count_black_and_white(window)
                    if white_p > self.MIN_PIX_AMOUNT:
                        primitive_img = cv2.imread(self.original_folder + pair['img'])
                        single_patch = patches_path + "/" + str(pair['id']) + "_patch_" + str(counter) + ".png"
                        cv2.imwrite(single_patch, primitive_img[y:y + winH, x:x + winW])
                        counter += 1

    def normal_patches(self):
        patches_path = os.path.join(self.patches_dir, "normal")
        winW, winH = self.patch_size[0], self.patch_size[1]
        if not os.path.exists(patches_path):
            os.mkdir(patches_path)
        normal_images = self.normal_images()
        for image_name in tqdm(normal_images):
            img_path = os.path.join(self.original_folder, image_name['img'])
            original_image = cv2.imread(img_path, 0)
            th2 = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
            counter = 0
            for (x, y, window) in self.sliding_window(th2, stepSize=112, windowSize=(winW, winH)):
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                white_p, black_p = self.count_black_and_white(window)
                if black_p > 100:
                    image_id = image_name['img'].split("_")[0]
                    single_patch = os.path.join(patches_path, "{}normal_{}.png".format(image_id, counter))
                    cv2.imwrite(single_patch, original_image[y: y + winH, x:x + winW])
                    counter += 1
                if counter > 10:
                    break

    def create_df_from_folder(self):
        class_dirs = os.listdir(self.patches_dir)
        image_df = {"IMG_PATH": list(), "Label": list()}
        for class_dir in class_dirs:
            if class_dir == 'normal':
                label = 0
            elif "Mass" in class_dir:
                label = 1
            else:
                label = 2
            class_images_p = os.path.join(self.patches_dir, class_dir)
            for image in os.listdir(class_images_p):
                image = os.path.join(class_dir, image)
                image_df['IMG_PATH'].append(image)
                image_df["Label"].append(label)
        pd.DataFrame(image_df, columns=["IMG_PATH", "Label"]).to_csv("INBPatches.csv")



if __name__ == "__main__":
    # print(len(os.listdir("patches_data/CalcificationSegmentationMasks")))
    # print(len(os.listdir("patches_data/MassSegmentationMasks")))
    # print(len(os.listdir("patches_data/normal")))
    slicer = Slicer("FullINbreast.csv",
                    [(1, "AllPNG/extras/MassSegmentationMasks", "Mask path"),
                     (2, "AllPNG/extras/CalcificationSegmentationMasks", "Calc path")],
                    "AllPNG/data/")
    slicer.create_df_from_folder()