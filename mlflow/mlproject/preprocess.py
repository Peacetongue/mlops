import os
import io
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import mlflow
import boto3
from botocore.client import Config

load_dotenv()

minio_client = boto3.client(
    's3',
    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4'),
    verify=False
)

BUCKET_NAME = "mlflow"


def download_from_minio(s3_path):
    buffer = io.BytesIO()
    minio_client.download_fileobj(BUCKET_NAME, s3_path, buffer)
    buffer.seek(0)
    return buffer


def upload_to_minio(file_object, s3_path):
    file_object.seek(0)
    minio_client.upload_fileobj(
        file_object,
        BUCKET_NAME,
        s3_path
    )


def preprocess_brats_data():
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("Brain_Tumor_Segmentation")

    with mlflow.start_run(run_name="Data_Preprocessing_MinIO_v2"):
        result = minio_client.list_objects(BUCKET_NAME, Prefix="raw_brats/", Delimiter='/')
        patient_folders = [prefix.get('Prefix') for prefix in result.get('CommonPrefixes', [])]

        mlflow.log_param("num_patients", len(patient_folders))
        scaler = MinMaxScaler()

        for patient_folder in patient_folders:
            patient_id = patient_folder.split('/')[-2]
            print(f"Processing patient: {patient_id}")

            patient_files = minio_client.list_objects(BUCKET_NAME, Prefix=patient_folder)
            file_keys = [obj['Key'] for obj in patient_files.get('Contents', [])]

            def find_file(pattern):
                return next((f for f in file_keys if pattern in f), None)

            t1_path = find_file('_t1.nii')
            t2_path = find_file('_t2.nii')
            t1ce_path = find_file('_t1ce.nii')
            flair_path = find_file('_flair.nii')
            seg_path = find_file('_seg.nii')

            if not all([t1_path, t2_path, t1ce_path, flair_path, seg_path]):
                print(f"Skipping patient {patient_id} - missing files")
                continue

            t1_img = nib.load(download_from_minio(t1_path)).get_fdata()
            t2_img = nib.load(download_from_minio(t2_path)).get_fdata()
            t1ce_img = nib.load(download_from_minio(t1ce_path)).get_fdata()
            flair_img = nib.load(download_from_minio(flair_path)).get_fdata()
            mask = nib.load(download_from_minio(seg_path)).get_fdata().astype(np.uint8)

            modalities = {
                't1': t1_img,
                't2': t2_img,
                't1ce': t1ce_img,
                'flair': flair_img
            }

            for mod_name, mod_data in modalities.items():
                modalities[mod_name] = scaler.fit_transform(
                    mod_data.reshape(-1, mod_data.shape[-1])).reshape(mod_data.shape)

            mask[mask == 4] = 3

            crop_slice = (slice(56, 184), slice(56, 184), slice(13, 141))
            combined_img = np.stack([
                modalities['flair'][crop_slice],
                modalities['t1ce'][crop_slice],
                modalities['t2'][crop_slice]
            ], axis=-1)

            mask_cropped = mask[crop_slice]

            img_buffer = io.BytesIO()
            mask_buffer = io.BytesIO()

            np.save(img_buffer, combined_img)
            np.save(mask_buffer, to_categorical(mask_cropped, num_classes=4))

            upload_to_minio(img_buffer, f"processed/images/{patient_id}.npy")
            upload_to_minio(mask_buffer, f"processed/masks/{patient_id}.npy")

            mlflow.log_metric("patients_processed", 1, step=int(patient_id.split('_')[-1]))

        mlflow.log_param("processing_status", "completed")
        print("All patients processed successfully")


if __name__ == "__main__":
    preprocess_brats_data()