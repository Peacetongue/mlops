import os
import numpy as np
import nibabel as nib
import glob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import splitfolders
from tqdm import tqdm


def run_preprocessing(input_dir: str, output_dir: str, final_dataset_dir: str):
    scaler = MinMaxScaler()

    t1_list = sorted(glob.glob(os.path.join(input_dir, '*', '*t1.nii.gz')))
    t2_list = sorted(glob.glob(os.path.join(input_dir, '*', '*t2.nii.gz')))
    t1ce_list = sorted(glob.glob(os.path.join(input_dir, '*', '*t1ce.nii.gz')))
    flair_list = sorted(glob.glob(os.path.join(input_dir, '*', '*flair.nii.gz')))
    mask_list = sorted(glob.glob(os.path.join(input_dir, '*', '*seg.nii.gz')))

    assert all(len(lst) == len(t1_list) for lst in [t2_list, t1ce_list, flair_list, mask_list]), "Mismatch in file lists!"

    for img_idx in tqdm(range(len(t1_list)), desc="Preprocessing"):
        temp_image_t1 = nib.load(t1_list[img_idx]).get_fdata()
        temp_image_t2 = nib.load(t2_list[img_idx]).get_fdata()
        temp_image_t1ce = nib.load(t1ce_list[img_idx]).get_fdata()
        temp_image_flair = nib.load(flair_list[img_idx]).get_fdata()
        temp_mask = nib.load(mask_list[img_idx]).get_fdata().astype(np.uint8)

        # Scale each modality
        temp_image_t1 = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)
        temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
        temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
        temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

        # Fix labels
        temp_mask[temp_mask == 4] = 3

        # Combine channels
        combined_image = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1], axis=-1)

        # Crop
        combined_image = combined_image[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]

        val, counts = np.unique(temp_mask, return_counts=True)
        if len(counts) > 1 and (1 - (counts[0] / counts.sum())) > 0.01:
            temp_mask_cat = to_categorical(temp_mask, num_classes=4)
            np.save(os.path.join(output_dir, "images", f"image_{img_idx}.npy"), combined_image)
            np.save(os.path.join(output_dir, "masks", f"mask_{img_idx}.npy"), temp_mask_cat)

    # Split folders
    print("Splitting into train/val...")
    splitfolders.ratio(output_dir, output=final_dataset_dir, seed=42, ratio=(.9, .1))

