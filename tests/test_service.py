"""Tests for the parts of the PIA service that don't need a GPU or a broker"""

import pytest

pytest.importorskip("workflows")
pytest.importorskip("ffs.index")

from ffs.service import (  # noqa: E402
    DetectorGeometry,
    Material,
    PiaRequest,
    read_cbf_header,
)


def test_read_cbf_header(dials_data):
    d = dials_data("cunir_serial", pathlib=True)
    header = read_cbf_header(d / "merlin0047_17000.cbf")
    assert header.detector_type == "Pilatus6M"
    assert header.wavelength == pytest.approx(1.37611)
    assert header.detector_parameters == {
        "pixel_size_x": pytest.approx(0.172),
        "pixel_size_y": pytest.approx(0.172),
        "material": Material.Si,
        "thickness": pytest.approx(0.45),
        "image_size_x": 2463,
        "image_size_y": 2527,
    }
    # The header values must be accepted by the detector model
    geometry = DetectorGeometry(
        distance=250,
        beam_center_x=217.3,
        beam_center_y=226.8,
        detector={
            "detector_type": header.detector_type,
            **header.detector_parameters,
        },
    )
    assert geometry.detector.image_size_y == 2527


def test_read_cbf_header_rejects_non_cbf(tmp_path):
    not_cbf = tmp_path / "image.cbf"
    not_cbf.write_text("not a CBF")
    with pytest.raises(ValueError):
        read_cbf_header(not_cbf)


@pytest.mark.parametrize(
    "filename, is_cbf",
    [("/data/x_00001.cbf", True), ("/data/x.CBF", True), ("/data/x.nxs", False)],
)
def test_request_is_cbf(filename, is_cbf):
    request = PiaRequest(
        dcid=1,
        dcgid=1,
        filename=filename,
        message_index=0,
        number_of_frames=1,
        start_frame_index=0,
        xBeam=0,
        yBeam=0,
        detector_distance=100,
    )
    assert request.is_cbf == is_cbf
