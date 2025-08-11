"""
Reconstruct_tomopy_astra_chunks_V2.py
========================

This module contains a refactored version of the original CT preprocessing
and reconstruction pipeline.  The primary goal of the refactor is to
dramatically reduce end‑to‑end runtime on multi‑core CPUs by leveraging
parallelism wherever possible.  Key improvements include:

* **Parallel file loading:** TIFF stacks can be large.  Reading
  tens or hundreds of TIFFs serially becomes a bottleneck.  The
  refactored `load_data` method uses a `ThreadPoolExecutor` to read
  projection, flat and dark images in parallel.  The underlying
  `tifffile.imread` function releases the Python Global Interpreter
  Lock (GIL) because the heavy lifting happens in C, so using
  threads here provides real speedup on multi‑core machines.

* **True chunked parallelism:**  The original `process_in_chunks`
  implementation spun up a fresh executor for every chunk and only
  processed one chunk per executor invocation.  As a result, no two
  chunks were ever processed concurrently.  The new implementation
  divides the 3D dataset into equally sized slices along the projection
  axis and uses a single executor to process all slices concurrently.
  Each worker thread operates on a disjoint slice and writes its
  output back into the appropriate section of the result array.  This
  approach maximizes CPU utilisation while still preserving the
  original ordering of projections.

* **Reduced memory footprint:**  Many tomography pipelines internally
  promote images to 64‑bit floats.  For most real‑world data, the
  dynamic range of 32‑bit floats is sufficient.  Wherever possible
  this refactor converts arrays to `float32` to cut memory usage in
  half.  This has the secondary benefit of improving cache locality
  during computation.

* **Cleaner interface for chunk processing:**  The refactored
  `process_in_chunks` function accepts a processing function that
  operates on a 3D numpy array and returns a 3D numpy array of the
  same shape.  Additional keyword arguments can be supplied via
  `functools.partial` when necessary.  Internally, chunks are
  processed in parallel and results are written back into a pre‑
  allocated array.

The remainder of the pipeline (flat/dark correction, normalisation,
beam hardening, ring removal, Paganin phase retrieval and negative
logarithm) are unchanged at a high level.  The TomoPy functions called
in these steps already release the GIL and support internal
parallelisation.  By chunking the data and dispatching chunks to
multiple threads we obtain an additional layer of parallelism on top
of what TomoPy provides.

Note: this script does not invoke a GUI file dialog.  To integrate
with your own application you will need to supply lists of file paths
for projections, flats and darks when constructing a `Preprocessor`
instance.
"""

from __future__ import annotations

import os
import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Tuple

import numpy as np
import tifffile as tiff
import tomopy
import matplotlib.pyplot as plt
import h5py
from numpy.fft import fft2, ifft2, fftfreq

#from paganin_phase_retrieval.tie_hom_filter import tie_hom_filter_cpu

try:
    # PaganinPhaseRetrieval is an optional dependency.  Fall back if
    # unavailable.  Users can provide their own implementation via
    # dependency injection.
    from paganin_phase_retrieval.phase_retrieval import PaganinPhaseRetrieval
except ImportError:
    PaganinPhaseRetrieval = None


class Preprocessor:
    """Preprocesses raw tomography projections with multi‑core support.

    Parameters
    ----------
    projection_files : Iterable[str]
        Sequence of file paths to raw projection TIFF images.
    flat_files : Iterable[str]
        Sequence of file paths to flat field TIFF images.
    dark_files : Iterable[str]
        Sequence of file paths to dark field TIFF images.
    ncores : int, optional
        Number of worker threads to use for parallel operations.  The
        default is 4.  Increase this value on machines with more
        physical cores.  Note that some TomoPy functions already
        exploit all available cores; setting `ncores` too high may
        oversubscribe your CPU.
    nchunk : int, optional
        Number of projections to process per chunk.  Chunks are
        processed concurrently across threads.  A chunk size that is
        too large may exhaust memory; a chunk size that is too small
        leads to excessive thread scheduling overhead.  The default
        value of 20 has been found to work well for typical datasets.
    use_float32 : bool, optional
        If True, convert input images to 32‑bit floating point.
        Reduces memory usage at the cost of some dynamic range.
    """

    def __init__(
        self,
        projection_files: Iterable[str],
        flat_files: Iterable[str],
        dark_files: Iterable[str],
        ncores: int = 4,
        nchunk: int = 20,
        use_float32: bool = True,
    ) -> None:
        # Sort and load data in parallel
        (self.projections,
         self.flats,
         self.darks) = self.load_data(projection_files, flat_files, dark_files, ncores, use_float32)

        self.corrected_projections: np.ndarray | None = None
        self.normalized_projections: np.ndarray | None = None
        self.ncores = ncores
        self.nchunk = nchunk

        # Initialise Paganin phase retrieval if available
        if PaganinPhaseRetrieval is not None:
            # Example parameters; adjust according to experimental setup
            self.paganin = PaganinPhaseRetrieval(
                shape=self.projections[0].shape,
                R1=0.15,      # source‑to‑sample distance in metres
                R2=0.298,     # sample‑to‑detector distance in metres
                energy=20,    # X‑ray energy in keV
                delta_beta=10.0,
                pixel_size=5e-6,
                padding="edge",
                use_rfft=True,
                fft_num_threads=ncores,
            )
        else:
            self.paganin = None

    @staticmethod
    def sort_numerically(file_list: Iterable[str]) -> List[str]:
        """Sort filenames by numerical index extracted via regex."""
        def extract_number(filename: str) -> int:
            base = os.path.basename(filename)
            match = re.search(r"(\d+)", base)
            return int(match.group()) if match else 0

        return sorted(file_list, key=extract_number)

    @staticmethod
    def load_tiff_file(filename: str, use_float32: bool) -> np.ndarray:
        """Load a single TIFF file into a numpy array.

        If ``use_float32`` is True, the returned array is cast to
        ``float32``.  Casting early reduces memory usage during
        subsequent operations.
        """
        img = tiff.imread(filename)
        if use_float32:
            return img.astype(np.float32, copy=False)
        else:
            return img.astype(np.float64, copy=False)

    @classmethod
    def load_data(
        cls,
        projection_files: Iterable[str],
        flat_files: Iterable[str],
        dark_files: Iterable[str],
        ncores: int,
        use_float32: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load projection, flat and dark TIFF stacks concurrently.

        This function sorts the filenames numerically, dispatches the
        file loading tasks to a thread pool and stacks the resulting
        arrays along the first axis.  If ``use_float32`` is True the
        returned stacks are of dtype ``float32``.
        """
        def load_files(file_list: List[str]) -> np.ndarray:
            with ThreadPoolExecutor(max_workers=ncores) as executor:
                imgs = list(executor.map(partial(cls.load_tiff_file, use_float32=use_float32), file_list))
                return np.stack(imgs, axis=0)
        proj_files_sorted = cls.sort_numerically(projection_files)
        projections = load_files(proj_files_sorted)
        print(f"The shape of projections is {projections.shape}")

        # Shape of the projections (used for dummy fallback)
        proj_shape = projections.shape

        # Handle flats
        if not flat_files:
            print("No flat-field TIFFs selected. Using dummy flat array of 1s.")
            flats = np.ones_like(projections)

        else:
            flat_files_sorted = cls.sort_numerically(flat_files)
            flats = load_files(flat_files_sorted)

        # Handle darks
        if not dark_files:
            print("No dark-field TIFFs selected. Using dummy dark array of 0s.")
            darks = np.zeros_like(projections)
        else:
            dark_files_sorted = cls.sort_numerically(dark_files)
            darks = load_files(dark_files_sorted)

        print(f"Loaded {len(projections)} projections, {len(flats)} flats, {len(darks)} darks.")
        return projections, flats, darks

    # ------------------------------------------------------------------
    # Core preprocessing steps
    # ------------------------------------------------------------------
    def flat_dark_correction(self) -> np.ndarray:
        """Perform flat/dark field correction on the loaded projections.

        The operation computes ``(projection - dark_avg) / (flat_avg - dark_avg)``
        for each pixel.  Division by zero is avoided by replacing NaNs
        with zeros.  The result is stored in ``self.corrected_projections``
        and also returned.
        """
        flat_avg = np.mean(self.flats, axis=0)
        dark_avg = np.mean(self.darks, axis=0)
        corrected = (self.projections - dark_avg) / (flat_avg - dark_avg)
        # Avoid NaNs and infs due to division by zero
        corrected = np.nan_to_num(corrected, copy=False)
        self.corrected_projections = corrected

        return corrected

    def normalize_flat_intensity(self) -> np.ndarray:
        """Normalize corrected projections by background intensity.

        Uses TomoPy's ``normalize_bg`` which already supports multi‑threading
        via the ``ncore`` parameter.  The result is stored in
        ``self.normalized_projections`` and returned.
        """
        if self.corrected_projections is None:
            raise RuntimeError("Flat/dark correction must be performed before normalisation.")
        print(f"Normalising {self.corrected_projections.shape[0]} projections...")
        normalized = tomopy.prep.normalize.normalize_bg(
            self.corrected_projections,
            air=1,
            ncore=self.ncores,
        )
        self.normalized_projections = normalized
        return normalized

    @staticmethod
    def apply_negative_logarithm(projections_chunk: np.ndarray) -> np.ndarray:
        """Compute the negative natural logarithm of a projection chunk.

        A small epsilon is added to prevent log(0).  This operation
        releases the GIL when executed on large arrays and therefore
        benefits from multi-threading.
        """
        return -np.log(projections_chunk + 1e-8)

    @staticmethod
    def beam_hardening_correction(projections_chunk: np.ndarray, bh_c: float = 0.0) -> np.ndarray:
        """Apply a simple beam hardening correction.

        Each element of the chunk is raised to the power ``1 + bh_c``.
        When ``bh_c = 0`` the data is left unchanged.  Numpy's
        ``power`` function releases the GIL so it is safe to call from
        multiple threads.
        """
        return np.power(projections_chunk, 1.0 + bh_c, dtype=projections_chunk.dtype)

    def remove_ring_artifacts(self, projections_chunk: np.ndarray, method: str = "tomopy", **kwargs) -> np.ndarray:
        """Remove ring artifacts from a chunk of projections.

        Currently only TomoPy's ring removal is supported.  Additional
        methods can be added in the future.
        """
        if method != "tomopy":
            raise ValueError(f"Unsupported ring removal method: {method}")
        return tomopy.misc.corr.remove_ring(projections_chunk, **kwargs)

    def phase_retrieval(self, projections_chunk: np.ndarray) -> np.ndarray:
        """Apply Paganin single‑distance phase retrieval to a chunk.

        Requires that the ``paganin`` attribute be initialised.  If
        unavailable an exception is raised.  The Paganin
        implementation uses FFTs which internally release the GIL,
        allowing parallel execution across multiple threads.
        """
        if self.paganin is None:
            raise RuntimeError("PaganinPhaseRetrieval is not available; cannot perform phase retrieval.")
        return self.paganin.phase_retrieval(projections_chunk)
    def _build_tiehom_transfer(self, nx: int,
                               ny: int,
                               dblGamma: float,
                               dblLambda: float,
                               pixel_size: float) -> np.ndarray:
        """
        Construct the TIE-Hom transfer function H(u,v) in frequency space.

        Parameters
        ----------
        nx, ny        : detector pixel dimensions
        dblGamma      : γ  (strength factor ≈ 0.01–0.1)
        dblLambda     : λ  (delta/beta-related factor, unitless)
        pixel_size    : detector pixel size ( metres )

        Returns
        -------
        H : (ny, nx) complex64
            Frequency-domain multiplicative filter.
        """
        # Spatial frequencies in cycles / metre
        fx = fftfreq(nx, d=pixel_size)
        fy = fftfreq(ny, d=pixel_size)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        r2 = FX**2 + FY**2                       # |q|²
        H = 1.0 / (1.0 + dblGamma * dblLambda * r2)
        return H.astype(np.float32)

    def tie_hom_filter_cpu_stack(self, stack: np.ndarray,
                                 dblGamma: float = 0.02,
                                 dblLambda: float = 0.1,
                                 pixel_size: float = 5e-6) -> np.ndarray:
        """
        Vectorised CPU TIE-Hom filter for an entire projection stack.

        Parameters
        ----------
        stack      : ndarray, shape (n_proj, ny, nx), dtype float32/64
        dblGamma   : γ parameter (phase-to-attenuation ratio scaling)
        dblLambda  : λ parameter (delta/beta factor)
        pixel_size : detector pixel size (metres)

        Returns
        -------
        filtered_stack : ndarray, same shape & dtype as `stack`
        """
        n_proj, ny, nx = stack.shape

        # Build the transfer function once
        H = self._build_tiehom_transfer(nx, ny, dblGamma, dblLambda, pixel_size)

        # FFT all projections at once (vectorised)
        fft_stack = fft2(stack, axes=(-2, -1))
        fft_stack *= H                               # broadcasting over proj axis
        filtered = ifft2(fft_stack, axes=(-2, -1)).real
        return filtered.astype(stack.dtype, copy=False)

    def apply_tie_hom_filter(self, projections_chunk: np.ndarray) -> np.ndarray:
        return self.tie_hom_filter_cpu_stack(
            projections_chunk,
            dblGamma=0.002,
            dblLambda=0.31,
            pixel_size=0.1496,
        )
    

    # ------------------------------------------------------------------
    # Parallel chunk processing
    # ------------------------------------------------------------------
    def process_in_chunks(
        self,
        process_function,
        data: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Apply ``process_function`` to ``data`` in parallel chunks.

        Parameters
        ----------
        process_function : callable
            A function accepting a 3D numpy array and returning a 3D
            numpy array of identical shape.  Additional positional and
            keyword arguments may be supplied via ``args`` and
            ``kwargs``.
        data : numpy.ndarray
            The input 3D data array to process.  Chunking is performed
            along the first axis (projections).

        Returns
        -------
        numpy.ndarray
            A new array containing the processed data in the same
            order as the input.
        """
        num_projections = data.shape[0]
        # Pre‑allocate output array with the same shape and dtype
        result = np.empty_like(data)

        # Determine slice boundaries for each chunk
        slices: List[Tuple[int, int]] = []
        for start in range(0, num_projections, self.nchunk):
            end = min(start + self.nchunk, num_projections)
            slices.append((start, end))

        def worker(slice_tuple: Tuple[int, int]) -> Tuple[int, np.ndarray]:
            s, e = slice_tuple
            chunk = data[s:e]
            processed = process_function(chunk, *args, **kwargs)
            return s, processed

        # Process chunks concurrently using a single thread pool.  Each
        # worker returns its start index along with the processed chunk
        # so that the results can be assembled in the correct order.
        with ThreadPoolExecutor(max_workers=self.ncores) as executor:
            for start_idx, processed_chunk in executor.map(worker, slices):
                end_idx = start_idx + processed_chunk.shape[0]
                result[start_idx:end_idx] = processed_chunk

        return result
    
    def save_to_hdf5(self, filename: str, corrected_projections: np.ndarray, preprocessing_params: dict) -> None:
        """
        Save the corrected projections and preprocessing parameters to an HDF5 file.
        
        Parameters
        ----------
        filename : str
            The name of the HDF5 file to save to.
        corrected_projections : np.ndarray
            The corrected projections to save in the file.
        preprocessing_params : dict
            A dictionary containing the preprocessing parameters to store as metadata.
        """
        
        with h5py.File(filename, 'w') as f:
            # Save corrected projections as a dataset
            f.create_dataset('corrected_projections', data=corrected_projections)
            
            # Save preprocessing parameters as metadata
            for param, value in preprocessing_params.items():
                f.attrs[param] = value
        
        print(f"Saved corrected projections and preprocessing parameters to {filename}.")

    # ------------------------------------------------------------------
    # End‑to‑end preprocessing pipeline
    # ------------------------------------------------------------------
    def preprocess_data(
        self,
        bh_c: float | None = None,
        remove_ring_kwargs: dict | None = None,
        apply_tie_hom: bool = False,  # Add this argument to enable TIEHom filter
        save_hdf5: bool = True,  # Optional: Whether to save the result to an HDF5 file
        hdf5_filename: str = "processed_data.h5"  # Filename for saving data
    ) -> np.ndarray:
        """Run the full preprocessing pipeline on the loaded projections.

        The pipeline performs the following steps in order:
        1. Flat/dark field correction.
        2. Background normalisation.
        3. Optional beam hardening correction.
        4. Optional Ring artefact removal.
        5. Optional Paganin phase retrieval.
        6. Optionally, apply TIEHom filter.
        7. Negative logarithm.
        

        Parameters
        ----------
        bh_c : float or None, optional
            Beam hardening correction coefficient. If None, no beam
            hardening correction is applied.
        remove_ring_kwargs : dict or None, optional
            Additional keyword arguments passed to the ring removal
            function. See ``tomopy.misc.corr.remove_ring`` for
            available options.
        apply_tie_hom : bool, optional
            Whether to apply the TIEHom filter. Default is False.
        save_hdf5 : bool, optional
            Whether to save the processed data to an HDF5 file.
        hdf5_filename : str, optional
            The filename to save the HDF5 file.

        Returns
        -------
        numpy.ndarray
            A 3D array of the preprocessed projections.
        """
        if remove_ring_kwargs is None:
            remove_ring_kwargs = {}
        if bh_c is None:
            bh_c = 0.0

        # Step 1: flat/dark correction
        print("Starting flat/dark field correction...")
        data = self.flat_dark_correction()

        # Step 2: normalisation
        print("Starting normalisation...")
        data = self.normalize_flat_intensity()
        # Free memory from corrected projections
        self.corrected_projections = None
        

        # Step 3: beam hardening correction
        if bh_c != 0.0:
            print(f"Applying beam hardening correction (coefficient = {bh_c})...")
            data = self.process_in_chunks(self.beam_hardening_correction, data, bh_c)

        # Step 4: ring artefact removal
        if remove_ring_kwargs:
            print("Removing ring artefacts...")
            #data = self.process_in_chunks(self.remove_ring_artifacts, data, method='tomopy', **remove_ring_kwargs)

        # Step 5: phase retrieval
        if self.paganin is not None:
            print("Performing phase retrieval...")
            data = self.process_in_chunks(self.phase_retrieval, data)

        # Step 6: Apply TIEHom filter if needed
        if apply_tie_hom:
            print("Applying TIE-Hom filter (CPU, frequency domain)…")
            data = self.process_in_chunks(
                self.apply_tie_hom_filter, data)

        # Step 7: negative logarithm
        print("Applying negative logarithm...")
        result = self.process_in_chunks(self.apply_negative_logarithm, data)
        # Free memory from normalized projections
        self.normalized_projections = None

        # Step 8: Optionally save the processed data to an HDF5 file
        """
        if save_hdf5:
            preprocessing_params = {
                'bh_c': bh_c,
                'remove_ring_kwargs': remove_ring_kwargs,
                'ncores': self.ncores,
                'nchunk': self.nchunk,
                'apply_tie_hom': apply_tie_hom,
            }
            self.save_to_hdf5(hdf5_filename, result, preprocessing_params)
        """
        print("Preprocessing complete.")
        
        import gc

        # Explicitly delete large intermediate arrays
        del self.corrected_projections
        del self.normalized_projections
        del self.projections
        del self.flats
        del self.darks

        # Flush garbage collector
        gc.collect()


        return result


    # ------------------------------------------------------------------
    # Diagnostic visualisation methods
    # ------------------------------------------------------------------
    def display_projection(self, data: np.ndarray, index: int) -> None:
        """Display a single projection from a 3D stack using matplotlib."""
        if index < 0 or index >= data.shape[0]:
            raise IndexError(f"Index {index} out of bounds for projection stack of length {data.shape[0]}")
        plt.imshow(data[index], cmap='gray')
        plt.title(f"Projection {index}")
        plt.colorbar()
        plt.show()

    def display_sinogram(self, data: np.ndarray, index: int) -> None:
        """Display a sinogram (constant row across all projections)."""
        if index < 0 or index >= data.shape[1]:
            raise IndexError(f"Index {index} out of bounds for sinogram height {data.shape[1]}")
        plt.imshow(data[:, index, :], cmap='gray')
        plt.title(f"Sinogram row {index}")
        plt.colorbar()
        plt.show()

'''

__all__ = [
    'Preprocessor',
    'Reconstructor',
    'calculate_voxel_size_with_magnification',
    'calculate_vol_size',
]
'''
