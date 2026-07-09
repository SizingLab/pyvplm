"""Unit tests for pyvplm.addon.comsoladdon (COMSOL file interface)."""

import numpy
import pytest

from pyvplm.addon.comsoladdon import import_file, save_file
from pyvplm.core.definition import PositiveParameter, PositiveParameterSet


@pytest.fixture
def simple_parameter_set():
    x = PositiveParameter("x", [0.01, 1], "m", "length")
    y = PositiveParameter("y", [10, 100], "N", "force")
    return PositiveParameterSet(x, y)


class TestSaveFile:
    def test_writes_one_line_per_parameter(self, simple_parameter_set, tmp_path):
        doe_x = numpy.array([[0.02, 20.0], [0.5, 50.0]])
        file_name = str(tmp_path / "comsol_input")
        save_file(doe_x, file_name, simple_parameter_set, is_SI=True, test_mode=True)
        content = (tmp_path / "comsol_input.txt").read_text().splitlines()
        assert content[0] == "x 0.02,0.5"
        assert content[1] == "y 20.0,50.0"

    def test_out_of_bounds_warns(self, simple_parameter_set, tmp_path):
        doe_x = numpy.array([[5.0, 20.0]])  # x above upper bound
        file_name = str(tmp_path / "comsol_input")
        with pytest.warns(UserWarning):
            save_file(doe_x, file_name, simple_parameter_set, is_SI=True, test_mode=True)

    def test_dimension_mismatch_rejected(self, simple_parameter_set, tmp_path):
        doe_x = numpy.array([[0.02, 20.0, 1.0]])
        with pytest.raises(ValueError):
            save_file(doe_x, str(tmp_path / "f"), simple_parameter_set, is_SI=True, test_mode=True)

    def test_wrong_types_rejected(self, simple_parameter_set, tmp_path):
        with pytest.raises(TypeError):
            save_file(
                [[0.02, 20.0]], str(tmp_path / "f"), simple_parameter_set, True, test_mode=True
            )


class TestImportFile:
    def test_import_SI_output(self, simple_parameter_set, tmp_path):
        file_path = tmp_path / "comsol_output.txt"
        file_path.write_text(
            "% Model: dummy\n% x y z\n0.02 20.0 1.0\n0.5 50.0 2.0\n",
        )
        doe = import_file(str(tmp_path / "comsol_output"), simple_parameter_set, "SI")
        assert list(doe.columns)[:2] == ["x", "y"]
        assert doe["x"].tolist() == pytest.approx([0.02, 0.5])
        assert doe["y"].tolist() == pytest.approx([20.0, 50.0])

    def test_import_with_units_conversion(self, simple_parameter_set, tmp_path):
        file_path = tmp_path / "comsol_units.txt"
        file_path.write_text("% x (mm) y (N)\n20.0 20.0\n500.0 50.0\n")
        doe = import_file(str(tmp_path / "comsol_units"), simple_parameter_set, "from_file")
        assert doe["x"].tolist() == pytest.approx([0.02, 0.5])  # converted from mm

    def test_invalid_units_option(self, simple_parameter_set, tmp_path):
        with pytest.raises(ValueError):
            import_file(str(tmp_path / "missing"), simple_parameter_set, "bad_option")

    def test_missing_file(self, simple_parameter_set, tmp_path):
        with pytest.raises(SyntaxError):
            import_file(str(tmp_path / "missing"), simple_parameter_set, "SI")
