
class Reconstructor:
    """Wrapper around ASTRA and TomoPy for cone‑beam reconstruction."""

    def __init__(self, vol_size: int) -> None:
        self.vol_size = vol_size
        self.vol_geom = None
        self.proj_geom = None
        self.reconstruction: np.ndarray | None = None

    

    def setup_geometry(self, projections: np.ndarray) -> Tuple[dict, dict]:
        """Define ASTRA volume and projection geometry from input data."""
        num_projections, detector_height, detector_width = projections.shape
        self.vol_geom = astra.create_vol_geom(self.vol_size, self.vol_size, self.vol_size)
        # Use a cone beam geometry covering 360 degrees
        angles = np.linspace(0.0, 2.0 * np.pi, num_projections, endpoint=False)
        self.proj_geom = astra.create_proj_geom(
            'cone', 1.0, 1.0, detector_height, detector_width, angles
        )
        return self.vol_geom, self.proj_geom

    def correct_cor_manually(self, projections: np.ndarray, cor: float) -> np.ndarray:
        """Manually shift projections to correct for centre of rotation."""
        return tomopy.prep.align_centers(projections, cor)

    def correct_cor_ng_vo(self, projections: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate and correct the centre of rotation via the VO algorithm."""
        cor = tomopy.find_center_vo(projections)
        corrected = tomopy.prep.align_centers(projections, cor)
        return corrected, cor

    def reconstruct_fdk(self, projections: np.ndarray, use_cuda: bool = True) -> np.ndarray:
        """Perform 3D cone‑beam reconstruction using ASTRA's FDK algorithm."""
        if self.vol_geom is None or self.proj_geom is None:
            raise RuntimeError("Geometry must be set up before reconstruction.")
        proj_id = astra.data3d.create('-proj3d', self.proj_geom, projections)
        rec_id = astra.data3d.create('-vol3d', self.vol_geom)
        cfg = astra.astra_dict('TJF_FDK_CUDA' if use_cuda else 'TJF_FDK')
        cfg['ProjectionDataId'] = proj_id
        cfg['ReconstructionDataId'] = rec_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        self.reconstruction = astra.data3d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        return self.reconstruction


def calculate_voxel_size_with_magnification(
    pixel_size: float, R1: float, R2: float
) -> float:
    """Compute the effective voxel size accounting for geometric magnification."""
    M = (R1 + R2) / R1
    return pixel_size / M


def calculate_vol_size(detector_pixels: int, pixel_size: float, voxel_size: float) -> int:
    """Estimate the number of voxels along one edge of the reconstruction."""
    fov = detector_pixels * pixel_size
    return int(fov / voxel_size)
