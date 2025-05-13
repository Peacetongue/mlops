import argparse
import os
from preprocess_core import run_preprocessing
import mlflow



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to BRATS data")
    parser.add_argument("--output_dir", required=True, help="Path to save processed images and masks")
    parser.add_argument("--final_dataset_dir", required=True, help="Path to save split train/val dataset")

    args = parser.parse_args()

    os.makedirs(args.output_dir + "/images", exist_ok=True)
    os.makedirs(args.output_dir + "/masks", exist_ok=True)
    os.makedirs(args.final_dataset_dir, exist_ok=True)

    run_preprocessing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        final_dataset_dir=args.final_dataset_dir
    )


if __name__ == "__main__":
    main()
