# run_preprocess_and_recon.py
#from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import h5py
import numpy as np
from viewer import show_projections_keyboard, show_projections_slider
import glob
import os
from Reconstruct_tomopy_astra_chunks_V2 import Preprocessor#, Reconstructor

# ------------------------------------------------------------
# Parameters you are likely to tweak
# ------------------------------------------------------------
NCORES   = 10          
NCHUNK   = 20          
BH_C     = 0.01        # beam-hardening coefficient, 0 means no correction
RING_KW  = {"rwidth": 10}
SAVE_H5  = True        # save both processed projections + recon
PROJ_H5  = "processed_projections.h5"
RECON_H5 = "reconstruction.h5"
USE_CUDA = False       # set True if you have an NVIDIA GPU + ASTRA-CUDA
# ------------------------------------------------------------

def main() -> None:
    # -------- 1. Select raw TIFFs  ---------------
    """root = Tk(); root.withdraw()
    proj_paths = filedialog.askopenfilenames(
        title="Select projection TIFFs", filetypes=[("TIFF", "*.tif *.tiff")]
    )
    flat_paths = filedialog.askopenfilenames(
        title="Select flat-field TIFFs", filetypes=[("TIFF", "*.tif *.tiff")]
    )
    dark_paths = filedialog.askopenfilenames(
        title="Select dark-field TIFFs", filetypes=[("TIFF", "*.tif *.tiff")]
    )
    """
    # Set your directories
    proj_dir = "/home/kadar/ads-ct/CtRecon/tubeV3"
    flat_dir = "/home/kadar/ads-ct/CtRecon/tubeV3"
    dark_dir = "/home/kadar/ads-ct/CtRecon/tubeV3"

    proj_pattern = "*scan*.tif"
    flat_pattern = "i*.tif"
    dark_pattern = "d*.tif"

    # Collect files matching both extension and name pattern
    proj_paths = sorted(glob.glob(os.path.join(proj_dir, proj_pattern)))
    flat_paths = sorted(glob.glob(os.path.join(flat_dir, flat_pattern)))
    dark_paths = sorted(glob.glob(os.path.join(dark_dir, dark_pattern)))
    print("Directory exists:", os.path.isdir(proj_dir))


    if not proj_paths or not flat_paths or not dark_paths:
        if not proj_paths:
            print("No projection TIFFs selected. Exiting.")
            return
        if not flat_paths:
            print("No flat-field TIFFs selected. Using empty array for flats.")
            flat_paths = []
        if not dark_paths:
            print("No dark-field TIFFs selected. Using empty array for darks.")
            dark_paths = []


    
    # -------- 2. Pre-process -------------------------------------
    pre = Preprocessor(
        projection_files=proj_paths,
        flat_files=flat_paths,
        dark_files=dark_paths,
        ncores=NCORES,
        nchunk=NCHUNK,
    )
    processed = pre.preprocess_data(
        bh_c=BH_C,
        remove_ring_kwargs=RING_KW,
        apply_tie_hom=True,
    )

    #show_projections_keyboard(processed)
    show_projections_slider(processed)

    # Optionally save the processed sinogram stack
    if SAVE_H5:
        with h5py.File(PROJ_H5, "w") as f:
            f.create_dataset("Processed_projections", data=processed, compression="gzip")
        print(f"Processed projections saved to {PROJ_H5}")



    # -------- 3. Reconstruction ----------------------------------
    reconstructor = Reconstructor(vol_size=processed.shape[1])   # square detector
    reconstructor.setup_geometry(processed)

    # If you already know the centre of rotation (COR), apply it manually:
    # processed = reconstructor.correct_cor_manually(processed, cor=1234.5)

    # Or have TomoPy estimate it automatically:
    processed, cor = reconstructor.correct_cor_ng_vo(processed)
    print(f"Estimated COR = {cor:.2f}")

    volume = reconstructor.reconstruct_fdk(processed, use_cuda=USE_CUDA)
    print(f"Reconstruction shape: {volume.shape}")

    # -------- 4. Save / view the reconstruction -------------------
    if SAVE_H5:
        with h5py.File(RECON_H5, "w") as f:
            f.create_dataset("volume", data=volume, compression="gzip")
        print(f"Reconstruction saved to {RECON_H5}")

    # Show one axial slice
    mid = volume.shape[0] // 2
    plt.figure(); plt.imshow(volume[mid], cmap="gray"); plt.title(f"Axial slice {mid}"); plt.colorbar(); plt.show()

if __name__ == "__main__":
    main()
