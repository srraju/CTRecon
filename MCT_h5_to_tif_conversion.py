import os
import h5py
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor, as_completed


# Input HDF5 file
h5_filename = "C:/Raju/Work/TiAnnealed_Dwell10_High_White_25ms_Stitch2.h5"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# === Utility to save TIFFs ===
def save_tiff(image, path):
    tifffile.imwrite(path, image, compression="zlib")

# === Multithreaded image extractor ===
def extract_and_save_images(h5_file, dataset_path, output_subfolder, prefix, use_parallel=True):
    full_path = os.path.join(output_dir, output_subfolder)
    os.makedirs(full_path, exist_ok=True)

    try:
        data = h5_file[dataset_path]
    except KeyError:
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return

    n_images = data.shape[0]
    print(f"[INFO] Saving {n_images} images from '{dataset_path}' to '{output_subfolder}/'...")

    def save_one(i):
        img = data[i]
        save_tiff(img, os.path.join(full_path, f"{prefix}_{i:04d}.tiff"))

    if use_parallel:
        with ThreadPoolExecutor() as executor:
            list(executor.map(save_one, range(n_images)))
    else:
        for i in range(n_images):
            save_one(i)

    print(f"[INFO] Finished saving {n_images} images from '{dataset_path}'.")

# === Metadata extraction ===
def extract_metadata(h5_file, output_path):
    metadata = []

    def safe_read(path, description):
        try:
            val = h5_file[path][()]
            if isinstance(val, bytes):
                val = val.decode()
            metadata.append(f"{description}: {val}")
        except KeyError:
            metadata.append(f"{description}: Not found")

    # Sample Angles
    safe_read("EXPERIMENT/SCANS/00_00/SAMPLE/ANGLES", "Angles (deg)")

    # Scan arc
    safe_read("EXPERIMENT/GEOMETRY/SCAN_ARC", "Scan_arc")

    # Detector metadata
    detector_path = "EXPERIMENT/INSTRUMENT/DETECTOR"
    if detector_path in h5_file:
        metadata.append("[Detector Metadata]")
        for key in h5_file[detector_path]:
            try:
                val = h5_file[f"{detector_path}/{key}"][()]
                if isinstance(val, bytes):
                    val = val.decode()
                metadata.append(f"{key}: {val}")
            except Exception as e:
                metadata.append(f"{key}: Error - {str(e)}")
    else:
        metadata.append("Detector metadata: Not found")

    # Source Energy
    safe_read("EXPERIMENT/INSTRUMENT/SOURCE/ENERGY", "Source Energy")

    with open(output_path, "w") as f:
        f.write("\n".join(metadata))

    print(f"[INFO] Metadata saved to '{output_path}'")

# === Main execution ===
def main():
    with h5py.File(h5_filename, "r") as h5_file:
        extract_and_save_images(h5_file, "EXPERIMENT/SCANS/00_00/SAMPLE/DATA", "Projections", "proj")
        extract_and_save_images(h5_file, "EXPERIMENT/SCANS/00_00/DARK_FIELD/BEFORE/DATA", "DarkFields", "dark")
        extract_and_save_images(h5_file, "EXPERIMENT/SCANS/00_00/FLAT_FIELD/BEFORE/DATA", "FlatFields", "flat")

        metadata_path = os.path.join(output_dir, "metadata.txt")
        extract_metadata(h5_file, metadata_path)

if __name__ == "__main__":
    main()