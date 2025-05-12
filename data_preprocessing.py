import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_data_path = "E:/PyCharm projects/Diploma/data/MICCAI_BraTS2020_TrainingData"


t1_list = sorted(glob.glob(train_data_path + '/*/*t1.nii.gz'))
t2_list = sorted(glob.glob(train_data_path + '/*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob(train_data_path + '/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob(train_data_path + '/*/*flair.nii.gz'))
mask_list = sorted(glob.glob(train_data_path + '/*/*seg.nii.gz'))

print(len(t1_list))


for img in range(len(t1_list)):  # Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    temp_image_t1 = nib.load(t1_list[img]).get_fdata()
    temp_image_t1 = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(
        temp_image_t1.shape)

    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
        temp_image_t2.shape)

    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(
        temp_image_t1ce.shape)

    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
        temp_image_flair.shape)

    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3

    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    # cropping x, y, and z
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141] # Тут я вырезаю БУКВАЛЬНО чёрные участки в данных, там НЕТ информации
    temp_mask = temp_mask[56:184, 56:184, 13:141]

    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save("../data/input_data_3channels/images/image_" + str(img), temp_combined_images)
        np.save("../data/input_data_3channels/masks/mask_" + str(img), temp_mask)


import splitfolders

print("start to split folder")
input_folder = "../data/input_data_3channels/"
output_folder = "../datasets"
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.9, .1), group_prefix=None) # default values
print("folder splitted")
