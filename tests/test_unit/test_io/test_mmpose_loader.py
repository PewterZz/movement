"""Unit tests for the MMPose predictions loader."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from movement.io.load_poses import from_mmpose_file


def _make_predictions(n_frames=5, n_individuals=2, n_keypoints=17):
    """Create synthetic MMPose predictions JSON data."""
    predictions = []
    for frame in range(n_frames):
        for person in range(n_individuals):
            kps = [
                [
                    100.0 + frame * 10 + person * 50 + k * 5,
                    200.0 + frame * 5 + k * 3,
                    0.85 + k * 0.01,
                ]
                for k in range(n_keypoints)
            ]
            predictions.append(
                {
                    "frame_id": frame,
                    "track_id": person,
                    "keypoints": kps,
                }
            )
    return predictions


@pytest.fixture
def mmpose_json_file(tmp_path):
    """Write a synthetic MMPose predictions file and return its path."""
    predictions = _make_predictions()
    file_path = tmp_path / "predictions.json"
    file_path.write_text(json.dumps(predictions))
    return file_path


@pytest.fixture
def single_individual_file(tmp_path):
    """MMPose file with a single individual per frame."""
    predictions = _make_predictions(n_frames=10, n_individuals=1)
    file_path = tmp_path / "single.json"
    file_path.write_text(json.dumps(predictions))
    return file_path


class TestFromMmposeFile:
    """Tests for from_mmpose_file()."""

    def test_basic_loading(self, mmpose_json_file):
        """Load a standard 2-person COCO-17 predictions file."""
        ds = from_mmpose_file(mmpose_json_file, fps=30)
        assert isinstance(ds, xr.Dataset)
        assert "position" in ds.data_vars
        assert "confidence" in ds.data_vars
        assert ds.sizes["time"] == 5
        assert ds.sizes["space"] == 2
        assert ds.sizes["keypoints"] == 17
        assert ds.sizes["individuals"] == 2

    def test_fps_as_seconds(self, mmpose_json_file):
        """Time coordinates should be in seconds when fps is provided."""
        ds = from_mmpose_file(mmpose_json_file, fps=30)
        expected_times = np.arange(5) / 30.0
        np.testing.assert_allclose(
            ds.coords["time"].values, expected_times, atol=1e-6
        )

    def test_fps_none_uses_frame_numbers(self, mmpose_json_file):
        """Time coordinates should be frame numbers when fps is None."""
        ds = from_mmpose_file(mmpose_json_file, fps=None)
        expected_times = np.arange(5, dtype=float)
        np.testing.assert_allclose(ds.coords["time"].values, expected_times)

    def test_coco_17_keypoint_names(self, mmpose_json_file):
        """COCO-17 schema should produce the standard 17 keypoint names."""
        ds = from_mmpose_file(mmpose_json_file, keypoint_schema="coco_17")
        kp_names = list(ds.coords["keypoints"].values)
        assert len(kp_names) == 17
        assert kp_names[0] == "nose"
        assert kp_names[-1] == "right_ankle"

    def test_halpe_26_schema(self, tmp_path):
        """Halpe-26 schema should produce 26 keypoint names."""
        predictions = _make_predictions(n_keypoints=26)
        file_path = tmp_path / "halpe.json"
        file_path.write_text(json.dumps(predictions))
        ds = from_mmpose_file(file_path, keypoint_schema="halpe_26")
        assert ds.sizes["keypoints"] == 26
        kp_names = list(ds.coords["keypoints"].values)
        assert "right_heel" in kp_names

    def test_custom_keypoint_names(self, tmp_path):
        """Custom keypoint names passed as a list should be used."""
        custom_names = ["head", "neck", "hip"]
        predictions = _make_predictions(n_keypoints=3)
        file_path = tmp_path / "custom.json"
        file_path.write_text(json.dumps(predictions))
        ds = from_mmpose_file(file_path, keypoint_schema=custom_names)
        assert list(ds.coords["keypoints"].values) == custom_names

    def test_single_individual(self, single_individual_file):
        """Single-individual files should produce individuals dim of 1."""
        ds = from_mmpose_file(single_individual_file, fps=25)
        assert ds.sizes["individuals"] == 1
        assert ds.sizes["time"] == 10

    def test_position_values_correct(self, mmpose_json_file):
        """Position values should match the input keypoint coordinates."""
        ds = from_mmpose_file(mmpose_json_file, fps=30)
        # Frame 0, individual_0, nose: x=100, y=200
        nose = ds["position"].sel(
            time=0, individuals="individual_0", keypoints="nose"
        )
        np.testing.assert_allclose(nose.values, [100.0, 200.0])

    def test_confidence_values_correct(self, mmpose_json_file):
        """Confidence values should match the input scores."""
        ds = from_mmpose_file(mmpose_json_file, fps=30)
        nose_conf = ds["confidence"].sel(
            time=0, individuals="individual_0", keypoints="nose"
        )
        np.testing.assert_allclose(nose_conf.values, 0.85, atol=1e-6)

    def test_source_software_attribute(self, mmpose_json_file):
        """Dataset should have source_software='MMPose' attribute."""
        ds = from_mmpose_file(mmpose_json_file)
        assert ds.attrs["source_software"] == "MMPose"

    def test_source_file_attribute(self, mmpose_json_file):
        """Dataset should record the source file path."""
        ds = from_mmpose_file(mmpose_json_file)
        assert "source_file" in ds.attrs

    def test_file_not_found_raises(self, tmp_path):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            from_mmpose_file(tmp_path / "nonexistent.json")

    def test_wrong_extension_raises(self, tmp_path):
        """Non-JSON file should raise ValueError."""
        file_path = tmp_path / "data.csv"
        file_path.write_text("x,y,z")
        with pytest.raises(ValueError, match="Expected a .json file"):
            from_mmpose_file(file_path)

    def test_empty_file_raises(self, tmp_path):
        """Empty JSON list should raise ValueError."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("[]")
        with pytest.raises(ValueError, match="non-empty list"):
            from_mmpose_file(file_path)

    def test_unknown_schema_raises(self, mmpose_json_file):
        """Unknown schema name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown keypoint_schema"):
            from_mmpose_file(mmpose_json_file, keypoint_schema="invalid")
