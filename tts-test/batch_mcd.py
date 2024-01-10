from pymcd.mcd import Calculate_MCD
import os
import numpy as np


def batch_calculate_mcd(original_folder, generated_folder):
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    mcd_values = []

    original_files = sorted(os.listdir(original_folder))
    generated_files = sorted(os.listdir(generated_folder))

    for orig_file, gen_file in zip(original_files, generated_files):
        orig_path = os.path.join(original_folder, orig_file)
        gen_path = os.path.join(generated_folder, gen_file)

        mcd_value = mcd_toolbox.calculate_mcd(orig_path, gen_path)
        print(f"MCD value for {orig_file} and {gen_file}: {mcd_value}")
        mcd_values.append(mcd_value)

    mean_mcd = np.mean(mcd_values)
    variance_mcd = np.var(mcd_values)

    print(f"Mean MCD value: {mean_mcd}")
    print(f"Variance of MCD values: {variance_mcd}")

original_folder_path = './original_data'
generated_folder_path = './gen_data'


batch_calculate_mcd(original_folder_path, generated_folder_path)
