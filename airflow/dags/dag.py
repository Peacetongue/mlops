import datetime
import logging
import os
import tempfile
import io
from typing import Optional  # Add this import

import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

MINIO_BUCKET_NAME = 'mlops'
NEW_DATA_PREFIX = 'new_data/'
PREPROCESSED_DATA_PREFIX = 'preprocessed_data/'

MODALITY_SUFFIXES = {
    "t1": "t1.nii",
    "t2": "t2.nii",
    "t1ce": "t1ce.nii",
    "flair": "flair.nii",
    "seg": "seg.nii"
}

DEFAULT_ARGS = {
    'owner': 'pren',
    'start_date': datetime.datetime(2025, 4, 1),
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=5),
    'aws_conn_id': 'aws_default',
}

def upload_numpy_to_s3(s3_hook: S3Hook, bucket_name: str, s3_key: str, np_array: np.ndarray):
    """Uploads a NumPy array to an S3 bucket."""
    with io.BytesIO() as f_byte:
        np.save(f_byte, np_array)
        f_byte.seek(0)
        s3_hook.load_bytes(bytes_data=f_byte.read(), key=s3_key, bucket_name=bucket_name, replace=True)
    logging.info(f"Successfully uploaded {s3_key} to S3 bucket {bucket_name}")

def identify_new_patient_data_fn(ti, bucket_name: str, new_prefix: str, processed_prefix: str):
    """
    Identifies patient IDs in the new_data prefix that are not yet in the processed_data prefix.
    Assumes patient data is stored in "folders" like new_prefix/patient_id/
    Returns a list of dictionaries, where each dictionary is suitable for expand_kwargs.
    Each dictionary should be like: {'op_kwargs': {'patient_id': 'id1', 'bucket_name': B, ...}}
    """
    s3_hook = S3Hook()
    
    new_patient_folders = s3_hook.list_prefixes(bucket_name=bucket_name, prefix=new_prefix, delimiter='/')
    if not new_patient_folders: new_patient_folders = []
    logging.info(f"Found potential new patient folders: {new_patient_folders}")

    processed_patient_folders = s3_hook.list_prefixes(bucket_name=bucket_name, prefix=processed_prefix, delimiter='/')
    if not processed_patient_folders: processed_patient_folders = []
    logging.info(f"Found processed patient folders: {processed_patient_folders}")

    def extract_patient_id(s3_folder_prefix: str, base_folder_prefix: str) -> Optional[str]:  # Changed type hint
        """Extracts patient_id from a full S3 prefix (e.g., new_data/patient_X/ -> patient_X)."""
        if s3_folder_prefix.startswith(base_folder_prefix):
            return s3_folder_prefix[len(base_folder_prefix):].strip('/')
        return None

    new_patient_ids = {extract_patient_id(p, new_prefix) for p in new_patient_folders}
    new_patient_ids = {pid for pid in new_patient_ids if pid}

    processed_patient_ids = {extract_patient_id(p, processed_prefix) for p in processed_patient_folders}
    processed_patient_ids = {pid for pid in processed_patient_ids if pid}

    patient_ids_to_process = list(new_patient_ids - processed_patient_ids)
    logging.info(f"Patient IDs to process: {patient_ids_to_process}")

    # Format for expand_kwargs:
    # A list of dicts, where each dict provides kwargs for one PythonOperator instance.
    # The PythonOperator needs its callable's args under the 'op_kwargs' key.
    list_of_kwargs_for_mapped_tasks = [
        {
            'op_kwargs': {  # This inner dict becomes the op_kwargs for preprocess_patient_data_fn
                'patient_id': pid,
                'bucket_name': bucket_name,
                'new_prefix': new_prefix,
                'processed_prefix': processed_prefix,
            }
        }
        for pid in patient_ids_to_process
    ]
    logging.info(f"Formatted kwargs for expand_kwargs: {list_of_kwargs_for_mapped_tasks}")
    
    # Using a more descriptive XCom key
    ti.xcom_push(key='preprocess_task_expand_kwargs', value=list_of_kwargs_for_mapped_tasks)
    return list_of_kwargs_for_mapped_tasks

def preprocess_patient_data_fn(bucket_name: str, new_prefix: str, processed_prefix: str, patient_id: str, **kwargs):
    """
    Preprocesses data for a single patient: downloads files from S3, processes them,
    and uploads results back to S3.
    """
    s3_hook = S3Hook()
    scaler = MinMaxScaler()

    logging.info(f"Starting preprocessing for patient: {patient_id}")

    patient_raw_data_root = f"{new_prefix}{patient_id}/"
    patient_processed_data_root = f"{processed_prefix}{patient_id}/"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_files = {}
        try:
            # List all S3 keys in the patient's raw data directory once
            all_s3_keys_in_patient_folder = s3_hook.list_keys(
                bucket_name=bucket_name, 
                prefix=patient_raw_data_root
            )
            # Ensure it's a list, even if empty or None was returned
            if not all_s3_keys_in_patient_folder:
                all_s3_keys_in_patient_folder = []
            
            logging.info(f"Files found under S3 prefix {patient_raw_data_root}: {all_s3_keys_in_patient_folder}")

            for modality, modality_suffix_pattern in MODALITY_SUFFIXES.items():
                
                found_s3_key_for_modality = None
                
                # Priority 1: Check for filename pattern: {patient_id}_{modality_suffix_pattern}
                for s3_key_candidate in all_s3_keys_in_patient_folder:
                    if not s3_key_candidate.startswith(patient_raw_data_root): continue # Should be redundant if prefix search is accurate
                    file_basename = s3_key_candidate.split('/')[-1]
                    if file_basename == f"{patient_id}_{modality_suffix_pattern}":
                        found_s3_key_for_modality = s3_key_candidate
                        logging.info(f"Found P1 match for {modality}: {found_s3_key_for_modality}")
                        break
                
                # Priority 2: If not found, check for exact filename: {modality_suffix_pattern}
                if not found_s3_key_for_modality:
                    for s3_key_candidate in all_s3_keys_in_patient_folder:
                        if not s3_key_candidate.startswith(patient_raw_data_root): continue
                        file_basename = s3_key_candidate.split('/')[-1]
                        if file_basename == modality_suffix_pattern:
                            found_s3_key_for_modality = s3_key_candidate
                            logging.info(f"Found P2 match for {modality}: {found_s3_key_for_modality}")
                            break
                
                # Priority 3: If not found, check for any file ending with {modality_suffix_pattern}
                if not found_s3_key_for_modality:
                    candidate_keys_p3 = []
                    for s3_key_candidate in all_s3_keys_in_patient_folder:
                        if not s3_key_candidate.startswith(patient_raw_data_root): continue
                        file_basename = s3_key_candidate.split('/')[-1]
                        if file_basename.endswith(modality_suffix_pattern):
                            candidate_keys_p3.append(s3_key_candidate)
                    
                    if candidate_keys_p3:
                        if len(candidate_keys_p3) > 1:
                            logging.warning(
                                f"Multiple generic files found for modality {modality} "
                                f"ending with '{modality_suffix_pattern}' in {patient_raw_data_root}: "
                                f"{candidate_keys_p3}. Using the first one: {candidate_keys_p3[0]}."
                            )
                        found_s3_key_for_modality = candidate_keys_p3[0]
                        logging.info(f"Found P3 match for {modality}: {found_s3_key_for_modality}")

                if not found_s3_key_for_modality:
                    error_msg = (
                        f"Required file for modality '{modality}' (expected to match patterns like "
                        f"'{patient_id}_{modality_suffix_pattern}', '{modality_suffix_pattern}', or "
                        f"ending with '{modality_suffix_pattern}') not found in S3 folder "
                        f"{patient_raw_data_root} for patient {patient_id}."
                    )
                    logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                # Define a consistent local filename for the downloaded file
                local_filename_for_download = f"{patient_id}_{modality_suffix_pattern}"
                desired_final_local_path = os.path.join(tmpdir, local_filename_for_download)

                logging.info(
                    f"Attempting to download {found_s3_key_for_modality} from bucket {bucket_name} "
                    f"into directory {tmpdir}. Target filename: {local_filename_for_download}"
                )
                
                # Download the file into tmpdir.
                actual_downloaded_path = s3_hook.download_file(
                    key=found_s3_key_for_modality, 
                    bucket_name=bucket_name, 
                    local_path=tmpdir,
                    preserve_file_name=False 
                )
                
                # Rename the downloaded file to our desired canonical name if it's different.
                if actual_downloaded_path != desired_final_local_path:
                    os.rename(actual_downloaded_path, desired_final_local_path)
                    logging.info(f"Renamed downloaded file from {actual_downloaded_path} to {desired_final_local_path}")
                else:
                    logging.info(f"Downloaded file already at desired path: {desired_final_local_path}")
                
                local_files[modality] = desired_final_local_path
                logging.info(f"Successfully prepared {found_s3_key_for_modality} as {local_files[modality]} in {tmpdir}")

            temp_image_t1 = nib.load(local_files['t1']).get_fdata()
            temp_image_t1 = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)

            temp_image_t2 = nib.load(local_files['t2']).get_fdata()
            temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

            temp_image_t1ce = nib.load(local_files['t1ce']).get_fdata()
            temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

            temp_image_flair = nib.load(local_files['flair']).get_fdata()
            temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

            temp_mask = nib.load(local_files['seg']).get_fdata()
            temp_mask = temp_mask.astype(np.uint8)
            temp_mask[temp_mask == 4] = 3

            temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

            x_start, x_end = 56, 184
            y_start, y_end = 56, 184
            z_start, z_end = 13, 141

            current_shape = temp_combined_images.shape
            
            actual_x_end = min(x_end, current_shape[0])
            actual_y_end = min(y_end, current_shape[1])
            actual_z_end = min(z_end, current_shape[2])
            
            if x_start >= actual_x_end or y_start >= actual_y_end or z_start >= actual_z_end:
                logging.warning(f"Patient {patient_id}: Image dimensions {current_shape} too small for effective crop with start indices ({x_start},{y_start},{z_start}). Skipping patient.")
                return f"Skipped_too_small_{patient_id}"

            temp_combined_images = temp_combined_images[x_start:actual_x_end, y_start:actual_y_end, z_start:actual_z_end]
            temp_mask = temp_mask[x_start:actual_x_end, y_start:actual_y_end, z_start:actual_z_end]


            val, counts = np.unique(temp_mask, return_counts=True)
            
            if counts.sum() == 0:
                logging.warning(f"Patient {patient_id}: Mask is empty after cropping. Skipping save.")
                return f"Skipped_empty_mask_{patient_id}"

            if (counts[0] / counts.sum()) < 0.99:
                logging.info(f"Patient {patient_id}: Meets useful volume criteria. Saving processed data.")
                
                num_classes = 4
                temp_mask_categorical = np.eye(num_classes, dtype=np.uint8)[temp_mask]

                image_s3_key = f"{patient_processed_data_root}image.npy" 
                mask_s3_key = f"{patient_processed_data_root}mask.npy"
                
                upload_numpy_to_s3(s3_hook, bucket_name, image_s3_key, temp_combined_images)
                upload_numpy_to_s3(s3_hook, bucket_name, mask_s3_key, temp_mask_categorical)
                
                logging.info(f"Successfully processed and saved data for patient: {patient_id}")
                return f"Success_{patient_id}"
            else:
                logging.info(f"Patient {patient_id}: Does not meet useful volume criteria (background ratio >= 0.99). Skipping save.")
                return f"Skipped_low_volume_{patient_id}"

        except Exception as e:
            logging.error(f"Error processing patient {patient_id}: {e}", exc_info=True)
            raise 

with DAG(
    dag_id='minio_mrt_preprocessing',
    default_args=DEFAULT_ARGS,
    schedule_interval='0 * * * *',
    max_active_runs=1, 
    catchup=False,
    tags=['preprocessing', 'minio', 's3'],
    doc_md="""MRT preprocessing DAG from MinIO"""
) as dag:

    identify_new_data_task = PythonOperator(
        task_id='identify_new_patient_data',
        python_callable=identify_new_patient_data_fn,
        op_kwargs={
            'bucket_name': MINIO_BUCKET_NAME,
            'new_prefix': NEW_DATA_PREFIX,
            'processed_prefix': PREPROCESSED_DATA_PREFIX,
        },
    )

    preprocess_data_task = PythonOperator.partial(
        task_id='preprocess_patient_data',
        python_callable=preprocess_patient_data_fn,
    ).expand_kwargs(
        identify_new_data_task.output  # This provides a list of dictionaries for op_kwargs
    )

    identify_new_data_task >> preprocess_data_task
