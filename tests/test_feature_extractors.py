import numpy as np
import pandas as pd
from implementation.feature_engineering.feature_extractors import (
    FeatureExtractor,
    get_percent_margin_point,
    get_line_from_points,
    get_gradient_angle,
    index_each_row
)


# ----- Helper functions -----

def _add_index(data: pd.DataFrame, index: list) -> None:
    "Adds an index in place"
    assert len(data) == len(index)
    assert "index" not in data.columns

    data['index'] = index
    data.set_index('index', drop=True, inplace=True)
    data.index.name = None


def _test_dataset_1():
    df = pd.DataFrame(
        [
            [-2, -1, 2, 3, 1, -1],
            [-15, -5, 5, 15, 5, -5]
        ],
        columns=[-2, -1, 0, 1, 2, 3]
    )
    _add_index(df, [298374, 12])
    return df


def _test_dataset_2():
    df = pd.DataFrame(
        [
            [1.0, 2.0, -5.0, 3.0, 10.0, -4.0],
            [100, 0, 12, 12, 12, 5],
            [-10, -2, 1, 2, 3, 4]
        ],
        columns=[-2, -1, 0, 1, 2, 3]
    )
    _add_index(df, [5, 12, 34])
    return df


def _test_feature_extractor(feature_name: str, input_dataset: pd.DataFrame, expected_dataset: pd.DataFrame):
    feature_extractor = FeatureExtractor([feature_name])
    feature_extractor.fit(input_dataset)
    output_dataset = feature_extractor.transform(input_dataset)
    pd.testing.assert_frame_equal(expected_dataset, output_dataset, check_less_precise=True)


# ----- Tests -----

def test_get_percent_margin_point():
    input_dataset = _test_dataset_1()
    # First Row: Max point is 3, half is 1.5, so left point will be -1 (index 1), and right point is 1 (index 4)
    # Second Row: Max point is 15, half is 7.5, so left point will be 5 (index 2), and right point is 5 (index 4)
    expected_left_index = pd.Series([1, 2], index=input_dataset.index)
    expected_right_index = pd.Series([4, 4], index=input_dataset.index)

    output_left_index, output_right_index = get_percent_margin_point(input_dataset)
    pd.testing.assert_series_equal(expected_left_index, output_left_index)
    pd.testing.assert_series_equal(expected_right_index, output_right_index)


def test_get_line_from_points():
    input_point_1 = (-2.0, 1.0)
    input_point_2 = (8.0, 6.0)
    # Expected values, see example https://www.mathematik-oberstufe.de/analysis/lin/gerade2d-2punkte.html
    expected_m = 0.5
    expected_k = 2

    output_m, output_k = get_line_from_points(input_point_1, input_point_2)
    assert output_m == expected_m
    assert output_k == expected_k


def test_get_gradient_angle__positive_gradient():
    input_gradient = 2 / 3
    expected_angle = 33.7

    output_angle = get_gradient_angle(input_gradient)
    np.isclose(expected_angle, output_angle)


def test_get_gradient_angle__negative_gradient():
    input_gradient = -1 / 2
    expected_angle = 153.4

    output_angle = get_gradient_angle(input_gradient)
    np.isclose(expected_angle, output_angle)


def test_get_gradient_angle__zero_gradient():
    input_gradient = 0
    expected_angle = 0

    output_angle = get_gradient_angle(input_gradient)
    np.isclose(expected_angle, output_angle)


def test_index_each_row():
    input_data = _test_dataset_2()
    input_index = pd.Series([0, 2, 1], index=input_data.index)

    expected_output = pd.Series([1.0, 12.0, -2.0], index=input_data.index)

    output = index_each_row(input_data, input_index)
    print(output)
    print(expected_output)
    pd.testing.assert_series_equal(expected_output, output)


def test_positive_amplitude():
    feature_name = 'positive_amplitude'
    input_dataset = _test_dataset_1()
    expected_dataset = pd.DataFrame([[3], [15]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_negative_amplitude():
    feature_name = 'negative_amplitude'
    input_dataset = _test_dataset_1()
    expected_dataset = pd.DataFrame([[-2], [-15]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_positive_signal_energy():
    feature_name = 'positive_signal_energy'
    input_dataset = _test_dataset_1()
    expected_dataset = pd.DataFrame([[14], [275]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_negative_signal_energy():
    feature_name = 'negative_signal_energy'
    input_dataset = _test_dataset_1()
    expected_dataset = pd.DataFrame([[6], [275]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_left_spike_angle():
    feature_name = 'left_spike_angle'
    input_dataset = _test_dataset_1()
    expected_dataset = pd.DataFrame([[63.4349], [84.2894]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_right_spike_angle():
    feature_name = 'right_spike_angle'
    input_dataset = _test_dataset_1()
    expected_dataset = pd.DataFrame([[116.5650], [95.7105]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_spike_width():
    feature_name = 'spike_width'
    input_dataset = _test_dataset_1()
    expected_dataset = pd.DataFrame([[3.5], [3]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_neo_coefficient_min():
    feature_name = 'neo_coefficient_min'
    input_dataset = _test_dataset_2()
    expected_dataset = pd.DataFrame([[19.0], [-1200.0], [14.0]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)


def test_neo_coefficient_max():
    feature_name = 'neo_coefficient_max'
    input_dataset = _test_dataset_2()
    expected_dataset = pd.DataFrame([[112.0], [144.0], [1.0]], columns=[feature_name], index=input_dataset.index)
    _test_feature_extractor(feature_name=feature_name,
                            input_dataset=input_dataset,
                            expected_dataset=expected_dataset)
