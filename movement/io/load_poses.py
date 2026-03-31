"""Load pose tracking data from various frameworks into ``movement``."""

import warnings
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
import pandas as pd
import pynwb
import xarray as xr
from sleap_io.io.slp import read_labels
from sleap_io.model.labels import Labels

from movement.io.load import register_loader
from movement.utils.logging import logger
from movement.validators.datasets import ValidPosesInputs
from movement.validators.files import (
    ValidAniposeCSV,
    ValidDeepLabCutCSV,
    ValidDeepLabCutH5,
    ValidFile,
    ValidNWBFile,
    ValidSleapAnalysis,
    ValidSleapLabels,
)


def from_numpy(
    position_array: np.ndarray,
    confidence_array: np.ndarray | None = None,
    individual_names: list[str] | None = None,
    keypoint_names: list[str] | None = None,
    fps: float | None = None,
    source_software: str | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from NumPy arrays.

    Parameters
    ----------
    position_array
        Array of shape (n_frames, n_space, n_keypoints, n_individuals)
        containing the poses. It will be converted to a
        :class:`xarray.DataArray` object named "position".
    confidence_array
        Array of shape (n_frames, n_keypoints, n_individuals) containing
        the point-wise confidence scores. It will be converted to a
        :class:`xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    individual_names
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "id_0", "id_1", etc.
    keypoint_names
        List of unique names for the keypoints in the skeleton. If None
        (default), the keypoints will be named "keypoint_0", "keypoint_1",
        etc.
    fps
        Frames per second of the video. Defaults to None, in which case
        the time coordinates will be in frame numbers.
    source_software
        Name of the pose estimation software from which the data originate.
        Defaults to None.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Examples
    --------
    Create random position data for two individuals, ``Alice`` and ``Bob``,
    with three keypoints each: ``snout``, ``centre``, and ``tail_base``.
    These are tracked in 2D space over 100 frames, at 30 fps.
    The confidence scores are set to 1 for all points.

    >>> import numpy as np
    >>> from movement.io import load_poses
    >>> rng = np.random.default_rng(seed=42)
    >>> ds = load_poses.from_numpy(
    ...     position_array=rng.random((100, 2, 3, 2)),
    ...     confidence_array=np.ones((100, 3, 2)),
    ...     individual_names=["Alice", "Bob"],
    ...     keypoint_names=["snout", "centre", "tail_base"],
    ...     fps=30,
    ... )

    """
    valid_poses_inputs = ValidPosesInputs(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software=source_software,
    )
    return valid_poses_inputs.to_dataset()


def from_file(
    file: Path | str,
    source_software: Literal[
        "DeepLabCut",
        "SLEAP",
        "LightningPose",
        "Anipose",
        "NWB",
    ],
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from any supported file.

    .. deprecated:: 0.14.0
        This function is deprecated and will be removed in a future release.
        Use :func:`movement.io.load_dataset<movement.io.load.load_dataset>`
        instead.

    Parameters
    ----------
    file
        Path to the file containing predicted poses. The file format must
        be among those supported by the ``from_dlc_file()``,
        ``from_slp_file()`` or ``from_lp_file()`` functions. One of these
        these functions will be called internally, based on
        the value of ``source_software``.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs
        Additional keyword arguments to pass to the software-specific
        loading functions that are listed under "See Also".

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.


    See Also
    --------
    movement.io.load_poses.from_dlc_file
    movement.io.load_poses.from_sleap_file
    movement.io.load_poses.from_lp_file
    movement.io.load_poses.from_anipose_file
    movement.io.load_poses.from_nwb_file

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_file(
    ...     "path/to/file.h5", source_software="DeepLabCut", fps=30
    ... )

    """
    warnings.warn(
        "The function `movement.io.load_poses.from_file` is deprecated"
        " and will be removed in a future release. "
        "Please use `movement.io.load_dataset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if source_software == "DeepLabCut":
        return from_dlc_file(file, fps)
    elif source_software == "SLEAP":
        return from_sleap_file(file, fps)
    elif source_software == "LightningPose":
        return from_lp_file(file, fps)
    elif source_software == "Anipose":
        return from_anipose_file(file, fps, **kwargs)
    elif source_software == "NWB":
        if fps is not None:
            logger.warning(
                "The fps argument is ignored when loading from an NWB file. "
                "The frame rate will be directly read or estimated from "
                "metadata in the file."
            )
        return from_nwb_file(file, **kwargs)
    else:
        raise logger.error(
            ValueError(f"Unsupported source software: {source_software}")
        )


def from_dlc_style_df(
    df: pd.DataFrame,
    fps: float | None = None,
    source_software: Literal["DeepLabCut", "LightningPose"] = "DeepLabCut",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a DeepLabCut-style DataFrame.

    Parameters
    ----------
    df
        DataFrame containing the pose tracks and confidence scores. Must
        be formatted as in DeepLabCut output files (see Notes).
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
    source_software
        Name of the pose estimation software from which the data originate.
        Defaults to "DeepLabCut", but it can also be "LightningPose"
        (because they use the same DataFrame format).

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    The DataFrame must have a multi-index column with the following levels:
    "scorer", ("individuals"), "bodyparts", "coords".
    The "individuals" level may be omitted if there is only one individual
    in the video.
    The "coords" level contains either:

    - the spatial coordinates "x", "y", and "likelihood"
      (point-wise confidence scores), or
    - the spatial coordinates "x", "y", and "z" (3D poses estimated by
      `triangulating 2D poses from multiple DeepLabCut output files\
      <https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html>`__).

    The row index corresponds to the frame number.

    See Also
    --------
    movement.io.load_poses.from_dlc_file

    """
    # Read names of individuals and keypoints from the DataFrame
    if "individuals" in df.columns.names:
        individual_names = (
            df.columns.get_level_values("individuals").unique().to_list()
        )
    else:
        individual_names = ["individual_0"]
    keypoint_names = (
        df.columns.get_level_values("bodyparts").unique().to_list()
    )
    # Extract position (and confidence if present)
    coord_names = df.columns.get_level_values("coords").unique().to_list()
    n_coords = len(coord_names)
    tracks = (
        df.to_numpy()
        .reshape((-1, len(individual_names), len(keypoint_names), n_coords))
        .transpose(0, 3, 2, 1)
    )
    if "likelihood" in coord_names:  # Coords: ['x', 'y', 'likelihood']
        likelihood_index = coord_names.index("likelihood")
        confidence_array = tracks[:, likelihood_index, :, :]
        pos_idx = [j for j in range(n_coords) if j != likelihood_index]
        position_array = tracks[:, pos_idx, :, :]
    else:  # Coords: ['x', 'y', 'z']
        position_array = tracks
        confidence_array = None
    return from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software=source_software,
    )


@register_loader(
    "SLEAP", file_validators=[ValidSleapLabels, ValidSleapAnalysis]
)
def from_sleap_file(file: str | Path, fps: float | None = None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a SLEAP file.

    Parameters
    ----------
    file
        Path to the file containing the SLEAP predictions in .h5
        (analysis) format. Alternatively, a .slp (labels) file can
        also be supplied (but this feature is experimental, see Notes).
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    The SLEAP predictions are normally saved in .slp files, e.g.
    "v1.predictions.slp". An analysis file, suffixed with ".h5" can be exported
    from the .slp file, using either the command line tool `sleap-convert`
    (with the "--format analysis" option enabled) or the SLEAP GUI (Choose
    "Export Analysis HDF5…" from the "File" menu) [1]_. This is the
    preferred format for loading pose tracks from SLEAP into *movement*.

    You can also directly load the .slp file. However, if the file contains
    multiple videos, only the pose tracks from the first video will be loaded.
    If the file contains a mix of user-labelled and predicted instances, user
    labels are prioritised over predicted instances to mirror SLEAP's approach
    when exporting .h5 analysis files [2]_.

    *movement* expects the tracks to be assigned and proofread before loading
    them, meaning each track is interpreted as a single individual. If
    no tracks are found in the file, *movement* assumes that this is a
    single-individual track, and will assign a default individual name.
    If multiple instances without tracks are present in a frame, the last
    instance is selected [2]_.
    Follow the SLEAP guide for tracking and proofreading [3]_.

    References
    ----------
    .. [1] https://docs.sleap.ai/latest/tutorial/exporting-the-results/#analysis-hdf5
    .. [2] https://github.com/talmolab/sleap/blob/v1.3.3/sleap/info/write_tracking_h5.py#L59
    .. [3] https://docs.sleap.ai/latest/guides/tracking-and-proofreading/

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_sleap_file("path/to/file.analysis.h5", fps=30)

    """
    valid_file = cast("ValidFile", file)
    file_path = valid_file.file
    if file_path.suffix == ".h5":
        ds = _ds_from_sleap_analysis_file(file_path, fps=fps)
    else:  # file.suffix == ".slp"
        ds = _ds_from_sleap_labels_file(file_path, fps=fps)
    # Add metadata as attrs
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded pose tracks from {file_path}:\n{ds}")
    return ds


@register_loader("LightningPose", file_validators=[ValidDeepLabCutCSV])
def from_lp_file(file: str | Path, fps: float | None = None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a LightningPose file.

    Parameters
    ----------
    file
        Path to the file containing the predicted poses, in .csv format.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_lp_file("path/to/file.csv", fps=30)

    """
    valid_file = cast("ValidFile", file)
    ds = _ds_from_lp_or_dlc_file(
        valid_file=valid_file, source_software="LightningPose", fps=fps
    )
    n_individuals = ds.sizes.get("individuals", 1)
    if n_individuals > 1:
        raise logger.error(
            ValueError(
                "LightningPose only supports single-individual datasets, "
                f"but the loaded dataset has {n_individuals} individuals. "
                "Did you mean to load from a DeepLabCut file instead?"
            )
        )
    return ds


@register_loader(
    "DeepLabCut", file_validators=[ValidDeepLabCutH5, ValidDeepLabCutCSV]
)
def from_dlc_file(file: str | Path, fps: float | None = None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a DeepLabCut file.

    Parameters
    ----------
    file
        Path to the file containing the predicted poses, either in .h5
        or .csv format.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    See Also
    --------
    movement.io.load_poses.from_dlc_style_df

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_dlc_file("path/to/file.h5", fps=30)

    Notes
    -----
    In ``movement``, pose data can only be loaded if all individuals have
    the same set of keypoints (i.e., the same labeled body parts).
    While DeepLabCut supports assigning keypoints that are not shared across
    individuals (see the `DeepLabCut documentation for multi-animal projects
    <https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#b-configure-the-project>`_),
    this feature is not currently supported in ``movement``.

    """
    return _ds_from_lp_or_dlc_file(
        valid_file=cast("ValidFile", file),
        source_software="DeepLabCut",
        fps=fps,
    )


def from_multiview_files(
    file_dict: dict[str, Path | str],
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    fps: float | None = None,
) -> xr.Dataset:
    """Load and merge pose tracking data from multiple views (cameras).

    .. deprecated:: 0.14.0
        This function is deprecated and will be removed in a future release.
        Use :func:`movement.io.load_multiview_dataset<movement.io.\
        load.load_multiview_dataset>` instead.

    Parameters
    ----------
    file_dict
        A dict whose keys are the view names and values are the paths to load.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata, with an additional ``views`` dimension.

    """
    warnings.warn(
        "The function `movement.io.load_poses.from_multiview_files` is "
        "deprecated and will be removed in a future release. "
        "Please use `movement.io.load_multiview_dataset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    views_list = list(file_dict.keys())
    new_coord_views = xr.DataArray(views_list, dims="view")
    dataset_list = [
        from_file(f, source_software=source_software, fps=fps)
        for f in file_dict.values()
    ]
    return xr.concat(dataset_list, dim=new_coord_views)


def _ds_from_lp_or_dlc_file(
    valid_file: ValidFile,
    source_software: Literal["LightningPose", "DeepLabCut"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a LightningPose or DLC file.

    Parameters
    ----------
    valid_file
        The validated LightningPose or DeepLabCut file object.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    # Load the DeepLabCut poses into a DataFrame
    file_path = valid_file.file
    df = (
        _df_from_dlc_csv(valid_file)
        if isinstance(valid_file, ValidDeepLabCutCSV)
        else pd.DataFrame(pd.read_hdf(file_path, key="df_with_missing"))
        # pd.read_hdf does not always return a DataFrame but we assume it does
        # in this case (since we know what's in the "df_with_missing" dataset)
    )
    logger.debug(f"Loaded poses from {file_path} into a DataFrame.")
    # Convert the DataFrame to an xarray dataset
    ds = from_dlc_style_df(df=df, fps=fps, source_software=source_software)
    # Add metadata as attrs
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded pose tracks from {file_path}:\n{ds}")
    return ds


def _ds_from_sleap_analysis_file(file: Path, fps: float | None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a SLEAP analysis (.h5) file.

    Parameters
    ----------
    file
        Path to the SLEAP analysis file containing predicted pose tracks.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    with h5py.File(file, "r") as f:
        # Transpose to shape: (n_frames, n_space, n_keypoints, n_tracks)
        tracks = f["tracks"][:].transpose(3, 1, 2, 0)
        # Create an array of NaNs for the confidence scores
        scores = np.full(tracks.shape[:1] + tracks.shape[2:], np.nan)
        individual_names = [n.decode() for n in f["track_names"][:]] or None
        if individual_names is None:
            logger.warning(
                f"Could not find SLEAP Track in {file}. "
                "Assuming single-individual dataset and assigning "
                "default individual name."
            )
        # If present, read the point-wise scores,
        # and transpose to shape: (n_frames, n_keypoints, n_tracks)
        if "point_scores" in f:
            scores = f["point_scores"][:].T
        return from_numpy(
            position_array=tracks.astype(np.float32),
            confidence_array=scores.astype(np.float32),
            individual_names=individual_names,
            keypoint_names=[n.decode() for n in f["node_names"][:]],
            fps=fps,
            source_software="SLEAP",
        )


def _ds_from_sleap_labels_file(file: Path, fps: float | None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a SLEAP labels (.slp) file.

    Parameters
    ----------
    file
        Path to the SLEAP labels file containing predicted pose tracks.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    labels = read_labels(file.as_posix())
    tracks_with_scores = _sleap_labels_to_numpy(labels)
    individual_names = [track.name for track in labels.tracks] or None
    if individual_names is None:
        logger.warning(
            f"Could not find SLEAP Track in {file}. "
            "Assuming single-individual dataset and assigning "
            "default individual name."
        )
    return from_numpy(
        position_array=tracks_with_scores[:, :-1, :, :],
        confidence_array=tracks_with_scores[:, -1, :, :],
        individual_names=individual_names,
        keypoint_names=[kp.name for kp in labels.skeletons[0].nodes],
        fps=fps,
        source_software="SLEAP",
    )


def _sleap_labels_to_numpy(labels: Labels) -> np.ndarray:
    """Convert a SLEAP ``Labels`` object to a NumPy array.

    The output array contains pose tracks and point-wise confidence scores.

    Parameters
    ----------
    labels
        A SLEAP `Labels` object.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing pose tracks and confidence scores,
        with shape ``(n_frames, 3, n_nodes, n_tracks)``.

    Notes
    -----
    This function only considers SLEAP instances in the first
    video of the SLEAP `Labels` object. User-labelled instances are
    prioritised over predicted instances, mirroring SLEAP's approach
    when exporting .h5 analysis files [1]_.

    This function is adapted from `Labels.numpy()` from the
    `sleap_io` package [2]_.

    References
    ----------
    .. [1] https://github.com/talmolab/sleap/blob/v1.3.3/sleap/info/write_tracking_h5.py#L59
    .. [2] https://github.com/talmolab/sleap-io

    """
    # Select frames from the first video only
    lfs = [lf for lf in labels.labeled_frames if lf.video == labels.videos[0]]
    # Figure out frame index range
    frame_idxs = [lf.frame_idx for lf in lfs]
    first_frame = min(0, min(frame_idxs))
    last_frame = max(0, max(frame_idxs))

    n_tracks = len(labels.tracks) or 1  # If no tracks, assume 1 individual
    individuals = labels.tracks or [None]
    skeleton = labels.skeletons[-1]  # Assume project only uses last skeleton
    n_nodes = len(skeleton.nodes)
    n_frames = int(last_frame - first_frame + 1)
    tracks = np.full((n_frames, 3, n_nodes, n_tracks), np.nan, dtype="float32")

    for lf in lfs:
        i = int(lf.frame_idx - first_frame)
        user_instances = lf.user_instances
        predicted_instances = lf.predicted_instances
        for j, ind in enumerate(individuals):
            user_track_instances = [
                inst for inst in user_instances if inst.track == ind
            ]
            predicted_track_instances = [
                inst for inst in predicted_instances if inst.track == ind
            ]
            # Use user-labelled instance if available
            if user_track_instances:
                inst = user_track_instances[-1]
                tracks[i, ..., j] = np.hstack(
                    (inst.numpy(), np.full((n_nodes, 1), np.nan))
                ).T
            elif predicted_track_instances:
                inst = predicted_track_instances[-1]
                tracks[i, ..., j] = inst.numpy(scores=True).T
    return tracks


def _df_from_dlc_csv(valid_file: ValidDeepLabCutCSV) -> pd.DataFrame:
    """Create a DeepLabCut-style DataFrame from a .csv file.

    If poses are loaded from a DeepLabCut-style .csv file, the DataFrame
    lacks the multi-index columns that are present in the .h5 file. This
    function parses the .csv file to DataFrame with multi-index columns,
    i.e. the same format as in the .h5 file.

    Parameters
    ----------
    valid_file
        The validated DeepLabCut-style CSV file object.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style DataFrame with multi-index columns.

    """
    # Deliberately avoid using pd.read_csv with index_col=0 here
    # and instead set the index after reading the CSV,
    # as in cases where the first data row is empty (e.g. "0,,,,,"),
    # pandas will misinterpret that value as the index name instead of a row.
    level_names = valid_file.level_names
    df = pd.read_csv(
        valid_file.file,
        header=list(range(len(level_names))),
    )
    df = df.set_index(df.columns[0])
    df.index.name = None
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=level_names)  # type: ignore[arg-type]
    return df


def from_anipose_style_df(
    df: pd.DataFrame,
    fps: float | None = None,
    individual_name: str = "id_0",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an Anipose 3D dataframe.

    Parameters
    ----------
    df
        Anipose triangulation dataframe
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.
    individual_name
        Name of the individual, by default "id_0"

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    Reshape dataframe with columns keypoint1_x, keypoint1_y, keypoint1_z,
    keypoint1_score,keypoint2_x, keypoint2_y, keypoint2_z,
    keypoint2_score...to array of positions with dimensions
    time, space, keypoints, individuals, and array of confidence (from scores)
    with dimensions time, keypoints, individuals.

    """
    keypoint_names = sorted(
        list(
            set(
                [
                    col.rsplit("_", 1)[0]
                    for col in df.columns
                    if any(col.endswith(f"_{s}") for s in ["x", "y", "z"])
                ]
            )
        )
    )

    n_frames = len(df)
    n_keypoints = len(keypoint_names)

    # Initialize arrays and fill
    position_array = np.zeros(
        (n_frames, 3, n_keypoints, 1)
    )  # 1 for single individual
    confidence_array = np.zeros((n_frames, n_keypoints, 1))
    for i, kp in enumerate(keypoint_names):
        for j, coord in enumerate(["x", "y", "z"]):
            position_array[:, j, i, 0] = df[f"{kp}_{coord}"]
        confidence_array[:, i, 0] = df[f"{kp}_score"]

    individual_names = [individual_name]

    return from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        source_software="Anipose",
        fps=fps,
    )


@register_loader("Anipose", file_validators=[ValidAniposeCSV])
def from_anipose_file(
    file: str | Path,
    fps: float | None = None,
    individual_name: str = "id_0",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an Anipose 3D .csv file.

    Parameters
    ----------
    file
        Path to the Anipose triangulation .csv file
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.
    individual_name
        Name of the individual, by default "id_0"

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    We currently do not load all information, only x, y, z, and score
    (confidence) for each keypoint. Future versions will load n of cameras
    and error.

    """
    valid_file = cast("ValidFile", file)
    anipose_df = pd.read_csv(valid_file.file)
    return from_anipose_style_df(
        anipose_df, fps=fps, individual_name=individual_name
    )


@register_loader("NWB", file_validators=[ValidNWBFile])
def from_nwb_file(
    file: str | Path | pynwb.file.NWBFile,
    processing_module_key: str = "behavior",
    pose_estimation_key: str = "PoseEstimation",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an NWB file.

    The input can be a path to an NWB file on disk or a
    :class:`pynwb.file.NWBFile` object.
    The data will be extracted from the NWB file's
    :class:`pynwb.base.ProcessingModule`
    (specified by ``processing_module_key``) that contains the
    ``ndx_pose.PoseEstimation`` object (specified by ``pose_estimation_key``)
    formatted according to the ``ndx-pose`` NWB extension [1]_.

    Parameters
    ----------
    file
        Path to the NWB file on disk (ending in ".nwb"),
        or an NWBFile object.
    processing_module_key
        Name of the :class:`pynwb.base.ProcessingModule` in the NWB file that
        contains the pose estimation data. Default is "behavior".
    pose_estimation_key
        Name of the ``ndx_pose.PoseEstimation`` object in the processing
        module (specified by ``processing_module_key``).
        Default is "PoseEstimation".

    Returns
    -------
    xarray.Dataset
        A single-individual ``movement`` dataset containing the pose tracks,
        confidence scores, and associated metadata.

    References
    ----------
    .. [1] https://github.com/rly/ndx-pose

    Examples
    --------
    Open an NWB file and load pose tracks from the
    :class:`pynwb.file.NWBFile` object:

    >>> import pynwb
    >>> import xarray as xr
    >>> from movement.io import load_poses
    >>> with pynwb.NWBHDF5IO("path/to/file.nwb", mode="r") as io:
    ...     nwb_file = io.read()
    ...     ds = load_poses.from_nwb_file(nwb_file)

    Or, directly load pose tracks from an NWB file on disk:

    >>> ds = load_poses.from_nwb_file("path/to/file.nwb")

    Load two single-individual datasets from two NWB files and merge them
    into a multi-individual dataset:

    >>> ds_singles = [
    ...     load_poses.from_nwb_file(f) for f in ["id1.nwb", "id2.nwb"]
    ... ]
    >>> ds_multi = xr.merge(ds_singles)

    """
    valid_file = cast("ValidFile", file)
    file_path_or_nwbfile_obj = valid_file.file
    if isinstance(file_path_or_nwbfile_obj, Path):
        with pynwb.NWBHDF5IO(file_path_or_nwbfile_obj, mode="r") as io:
            nwbfile_object = io.read()
            ds = _ds_from_nwb_object(
                nwbfile_object, processing_module_key, pose_estimation_key
            )
            ds.attrs["source_file"] = file_path_or_nwbfile_obj
    else:  # file is an NWBFile object
        ds = _ds_from_nwb_object(
            file_path_or_nwbfile_obj,
            processing_module_key,
            pose_estimation_key,
        )
    return ds


def _ds_from_nwb_object(
    nwb_file: pynwb.file.NWBFile,
    processing_module_key: str = "behavior",
    pose_estimation_key: str = "PoseEstimation",
) -> xr.Dataset:
    """Extract a ``movement`` poses dataset from an NWBFile object.

    Parameters
    ----------
    nwb_file
        An NWBFile object.
    processing_module_key
        Name of the :class:`pynwb.base.ProcessingModule` in the NWB file that
        contains the pose estimation data. Default is "behavior".
    pose_estimation_key
        Name of the ``ndx_pose.PoseEstimation`` object in the processing
        module (specified by ``processing_module_key``).
        Default is "PoseEstimation".

    Returns
    -------
    xarray.Dataset
        A single-individual ``movement`` poses dataset

    """
    pose_estimation = nwb_file.processing[processing_module_key][
        pose_estimation_key
    ]
    source_software = pose_estimation.source_software
    pose_estimation_series = pose_estimation.pose_estimation_series
    single_keypoint_datasets = []
    for keypoint, pes in pose_estimation_series.items():
        # Extract position and confidence data for each keypoint
        position_data = np.asarray(pes.data)  # shape: (n_frames, n_space)
        confidence_data = (  # shape: (n_frames,)
            np.asarray(pes.confidence)
            if getattr(pes, "confidence", None) is not None
            else np.full(position_data.shape[0], np.nan)
        )
        # Compute fps from time differences between timestamps
        # if rate is not available
        fps = pes.rate or float(np.nanmedian(1 / np.diff(pes.timestamps)))
        single_keypoint_datasets.append(
            # create movement dataset with 1 keypoint and 1 individual
            from_numpy(
                position_data[:, :, np.newaxis, np.newaxis],
                confidence_data[:, np.newaxis, np.newaxis],
                individual_names=[nwb_file.identifier],
                keypoint_names=[keypoint],
                fps=round(fps, 6),
                source_software=source_software,
            )
        )
    return xr.merge(
        single_keypoint_datasets, join="outer", compat="no_conflicts"
    )


def from_mmpose_file(
    file: str | Path,
    fps: float | None = None,
    keypoint_schema: str = "coco_17",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an MMPose predictions file.

    MMPose saves predictions as a JSON file containing a list of instance
    objects. Each instance has ``keypoints`` (shape ``K x 3``: x, y,
    confidence) and ``frame_id`` identifying the video frame.

    Parameters
    ----------
    file
        Path to the MMPose predictions JSON file. The file is expected to
        contain a list of instance dicts, each with at minimum:
        ``frame_id`` (int), ``keypoints`` (list of [x, y, score] triplets),
        and optionally ``track_id`` (int, for multi-individual tracking).
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
    keypoint_schema
        Keypoint schema name used to assign keypoint names. Currently
        supported: ``"coco_17"`` (COCO 17-keypoint body pose, default),
        ``"coco_133"`` (COCO WholeBody), ``"halpe_26"``.
        Pass a list of strings to use custom keypoint names.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_mmpose_file("predictions.json", fps=30)

    """
    import json

    _KEYPOINT_SCHEMAS: dict[str, list[str]] = {
        "coco_17": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        "halpe_26": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "head",
            "neck",
            "hip",
            "left_big_toe",
            "right_big_toe",
            "left_small_toe",
            "right_small_toe",
            "left_heel",
            "right_heel",
        ],
    }

    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"MMPose predictions file not found: {file_path}")
    if file_path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected a .json file, got '{file_path.suffix}'. "
            "MMPose predictions should be saved with --out predictions.json."
        )

    with open(file_path) as f:
        instances = json.load(f)

    if not isinstance(instances, list) or len(instances) == 0:
        raise ValueError(
            f"Expected a non-empty list of instance dicts in {file_path}. "
            "Check that the file contains MMPose output."
        )

    # Resolve keypoint names
    if isinstance(keypoint_schema, list):
        keypoint_names = keypoint_schema
    elif keypoint_schema in _KEYPOINT_SCHEMAS:
        keypoint_names = _KEYPOINT_SCHEMAS[keypoint_schema]
    else:
        raise ValueError(
            f"Unknown keypoint_schema '{keypoint_schema}'. "
            f"Choose from {list(_KEYPOINT_SCHEMAS.keys())} or pass a list of names."
        )
    n_keypoints = len(keypoint_names)

    # Group instances by frame_id
    frames_dict: dict[int, list[dict]] = {}
    for inst in instances:
        fid = int(inst.get("frame_id", inst.get("image_id", 0)))
        frames_dict.setdefault(fid, []).append(inst)

    sorted_frame_ids = sorted(frames_dict.keys())
    n_frames = len(sorted_frame_ids)

    # Determine the number of individuals (max instances in any frame)
    max_individuals = max(len(insts) for insts in frames_dict.values())
    individual_names = [f"individual_{i}" for i in range(max_individuals)]

    # Arrays: (n_frames, n_space=2, n_keypoints, n_individuals)
    position_array = np.full(
        (n_frames, 2, n_keypoints, max_individuals), np.nan, dtype=np.float32
    )
    confidence_array = np.full(
        (n_frames, n_keypoints, max_individuals), np.nan, dtype=np.float32
    )

    for frame_idx, fid in enumerate(sorted_frame_ids):
        frame_instances = frames_dict[fid]
        for ind_idx, inst in enumerate(frame_instances):
            kps = inst.get("keypoints", [])
            if not kps:
                continue
            kp_array = np.array(kps, dtype=np.float32)
            # kp_array shape: (K, 3) — x, y, score
            # Clip to n_keypoints in case schema mismatch
            k = min(kp_array.shape[0], n_keypoints)
            position_array[frame_idx, 0, :k, ind_idx] = kp_array[:k, 0]  # x
            position_array[frame_idx, 1, :k, ind_idx] = kp_array[:k, 1]  # y
            confidence_array[frame_idx, :k, ind_idx] = kp_array[:k, 2]

    ds = from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software="MMPose",
    )
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded MMPose pose tracks from {file_path}:\n{ds}")
    return ds


def from_coco_file(
    file: str | Path,
    fps: float | None = None,
    keypoint_schema: str = "coco_17",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a COCO keypoint annotations file.

    COCO stores keypoints in the annotations JSON as a flat list of
    ``[x1, y1, v1, x2, y2, v2, ...]`` per annotation, where ``v`` is
    the visibility flag (0 = not labeled, 1 = labeled but occluded,
    2 = labeled and visible). This loader maps visibility to a
    confidence score: 0 -> NaN, 1 -> 0.5, 2 -> 1.0.

    Parameters
    ----------
    file
        Path to a COCO-format annotations JSON file. Must contain
        an ``annotations`` key with a list of annotation dicts, each
        having ``keypoints`` (flat list), ``image_id`` (int), and
        optionally ``id`` (annotation id).
    fps
        Frames per second. If None, ``time`` coordinates use image
        indices.
    keypoint_schema
        Keypoint schema name or a list of custom names. Supported
        built-in schemas: ``"coco_17"`` (default).

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset with pose tracks and confidence scores.

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_coco_file("annotations.json", fps=30)

    """
    import json

    _COCO_SCHEMAS: dict[str, list[str]] = {
        "coco_17": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
    }

    _VISIBILITY_TO_CONFIDENCE = {0: float("nan"), 1: 0.5, 2: 1.0}

    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(
            f"COCO annotations file not found: {file_path}"
        )
    if file_path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected a .json file, got '{file_path.suffix}'."
        )

    with open(file_path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "annotations" not in data:
        raise ValueError(
            f"Expected a COCO-format JSON with an 'annotations' key "
            f"in {file_path}."
        )

    annotations = data["annotations"]
    if not annotations:
        raise ValueError(
            f"No annotations found in {file_path}."
        )

    # Resolve keypoint names
    if isinstance(keypoint_schema, list):
        keypoint_names = keypoint_schema
    elif keypoint_schema in _COCO_SCHEMAS:
        keypoint_names = _COCO_SCHEMAS[keypoint_schema]
    else:
        raise ValueError(
            f"Unknown keypoint_schema '{keypoint_schema}'. "
            f"Choose from {list(_COCO_SCHEMAS.keys())} or pass a list."
        )
    n_keypoints = len(keypoint_names)

    # Group annotations by image_id (each image_id is one "frame")
    frames_dict: dict[int, list[dict]] = {}
    for ann in annotations:
        img_id = int(ann.get("image_id", 0))
        frames_dict.setdefault(img_id, []).append(ann)

    sorted_image_ids = sorted(frames_dict.keys())
    n_frames = len(sorted_image_ids)
    max_individuals = max(len(anns) for anns in frames_dict.values())
    individual_names = [f"individual_{i}" for i in range(max_individuals)]

    position_array = np.full(
        (n_frames, 2, n_keypoints, max_individuals), np.nan, dtype=np.float32
    )
    confidence_array = np.full(
        (n_frames, n_keypoints, max_individuals), np.nan, dtype=np.float32
    )

    for frame_idx, img_id in enumerate(sorted_image_ids):
        for ind_idx, ann in enumerate(frames_dict[img_id]):
            flat_kps = ann.get("keypoints", [])
            if not flat_kps:
                continue
            # COCO flat format: [x1, y1, v1, x2, y2, v2, ...]
            n_triplets = min(len(flat_kps) // 3, n_keypoints)
            for k in range(n_triplets):
                x = float(flat_kps[k * 3])
                y = float(flat_kps[k * 3 + 1])
                v = int(flat_kps[k * 3 + 2])
                position_array[frame_idx, 0, k, ind_idx] = x
                position_array[frame_idx, 1, k, ind_idx] = y
                confidence_array[frame_idx, k, ind_idx] = (
                    _VISIBILITY_TO_CONFIDENCE.get(v, float("nan"))
                )

    ds = from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software="COCO",
    )
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded COCO keypoint annotations from {file_path}:\n{ds}")
    return ds
