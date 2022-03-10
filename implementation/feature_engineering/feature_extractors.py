from abc import ABC, abstractmethod
from typing import Callable, Union, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ----- Helper functions -----

def get_percent_margin_point(data: pd.DataFrame, percentage=0.5) -> Tuple[pd.Series, pd.Series]:
    """
    Gets the index of the left and the right 50% margin
    :param data: the input signal data
    :param percentage: the percentage which should be used for calculating at which point the percentage is crossed
    :return: left_int_index, right_int_index
    """
    data = data.copy()
    # Quick hack to get integer index
    data.columns = [i for i in range(len(data.columns))]
    value_at_percentage = data.max(axis=1) * percentage
    idx_of_max_value = data.idxmax(axis=1)
    assert idx_of_max_value.nunique() == 1, "Unexpected values found, expected the same max value in each signal"
    max_index = idx_of_max_value.values[0]  # Can be done because of previous check

    data_minus_percentage_value = data.sub(value_at_percentage, axis=0)
    negative_or_zero_values = data_minus_percentage_value <= 0

    right_data = negative_or_zero_values.iloc[:, max_index:]
    right_int_index = right_data.idxmax(axis=1)

    left_data = negative_or_zero_values.iloc[:, :max_index + 1].iloc[:, ::-1]
    left_data = left_data.T.reset_index(drop=True).T
    left_int_index = idx_of_max_value.sub(left_data.idxmax(axis=1))

    return left_int_index, right_int_index


def get_line_from_points(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Simple geometry.
    Formula e.g.: https://www.mathematik-oberstufe.de/analysis/lin/gerade2d-2punkte.html
    :param point_1: One point in the line.
    :param point_2: One point in the line.
    :return: the slope m and the intercept k of the line m*x + k
    """
    point_1_x, point_1_y = point_1
    point_2_x, point_2_y = point_2

    # Check if the line is vertical
    if point_1_x == point_2_x:
        raise ValueError("Line cannot be vertical")

    m = (point_2_y - point_1_y) / (point_2_x - point_1_x)
    k = -m * point_1_x + point_1_y
    return m, k


def get_gradient_angle(gradient: float) -> float:
    """
    Calculates the gradient angle of a slope/line.
    Formula: https://www.mathematik-oberstufe.de/analysis/lin/gerade2d-steigungswinkel.html
    :param gradient: the gradient of the slope
    :return: the angle in degrees
    """
    if gradient == 0:
        return 0.0
    if gradient > 0:
        return np.degrees(np.arctan(gradient))
    if gradient < 0:
        return np.degrees(np.arctan(gradient) + np.pi)


def index_each_row(data: pd.DataFrame, input_index: pd.Series) -> pd.Series:
    """
    Per row, get the nth column defined by the index at that row.
    :param data: the input data
    :param input_index: the index, with the same length as the data
    :return: the indexed data
    """
    data = data.copy()
    assert len(data) == len(input_index)
    pd.testing.assert_index_equal(data.index, input_index.index)

    indexed_data = data.apply(lambda x: x.iloc[input_index.loc[x.name]], axis=1)
    return indexed_data


def get_lines_for_index(data: pd.DataFrame, index: pd.Series) -> Tuple[float, float]:
    """

    :param data: The signal data
    :param index: The index for which points the line should be calculated for each row
    :return: the line data, with columns `m` (slope) and `k` (intercept)
    """
    data = data.copy()

    points_data_dict = {}
    points_data_dict["left_point_x"] = index - 1
    points_data_dict["right_point_x"] = index + 1
    points_data_dict["left_point_y"] = index_each_row(data, points_data_dict["left_point_x"])
    points_data_dict["right_point_y"] = index_each_row(data, points_data_dict["right_point_x"])

    points_data = pd.DataFrame(points_data_dict)

    lines_data = points_data.apply(
        lambda row: get_line_from_points(
            (row['left_point_x'], row['left_point_y']),
            (row['right_point_x'], row['right_point_y']),
        ),
        axis=1,
        result_type="expand"
    )
    lines_data.columns = ['m', 'k']
    return lines_data


def calculate_neo_coefficient(data: pd.DataFrame, index: pd.Series) -> pd.Series:
    """
    :param data: The signal data
    :param index: The index for which points the line should be calculated for each row
    :return: the value of the neo coefficient at the requested index
    """
    data = data.copy()
    n = index_each_row(data, index)
    n_minus_1 = index_each_row(data, index - 1)
    n_plus_1 = index_each_row(data, index + 1)

    neo = n ** 2 - n_minus_1 * n_plus_1

    return neo


# ----- Feature extractors -----


class BaseFeatureExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        pass


class PositiveAmplitude(BaseFeatureExtractor):
    """
    The highest point of the signal
    """

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        return data.max(axis=1)

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'positive_amplitude'
        return data_transformed


class NegativeAmplitude(BaseFeatureExtractor):
    """
    The lowest point of the signal
    """

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        return data.min(axis=1)

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'negative_amplitude'
        return data_transformed


class PositiveSignalEnergy(BaseFeatureExtractor):
    """
    Signal energy of the positive part of the signal
    """

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        positive_data = data.copy()
        positive_data[positive_data < 0] = 0
        positive_data = positive_data ** 2
        return positive_data.sum(axis=1)

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'positive_signal_energy'
        return data_transformed


class NegativeSignalEnergy(BaseFeatureExtractor):
    """
    Signal energy of the negative part of the signal
    """

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        negative_data = data.copy()
        negative_data[negative_data > 0] = 0
        negative_data = negative_data ** 2
        return negative_data.sum(axis=1)

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'negative_signal_energy'
        return data_transformed


class LeftSpikeAngle(BaseFeatureExtractor):
    """
    The angle at the left spike.
    """

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        index, _ = get_percent_margin_point(data.iloc[:, 1:], percentage=0.5)
        index += 1

        line_data = get_lines_for_index(data, index)

        gradient_angle = line_data['m']
        gradient_angle = gradient_angle.apply(get_gradient_angle)

        return gradient_angle

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'left_spike_angle'
        return data_transformed


class RightSpikeAngle(BaseFeatureExtractor):
    """
    The angle at the right spike.
    """

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        _, index = get_percent_margin_point(data.iloc[:, :-1], percentage=0.5)
        line_data = get_lines_for_index(data, index)

        gradient_angle = line_data['m']
        gradient_angle = gradient_angle.apply(get_gradient_angle)

        return gradient_angle

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'right_spike_angle'
        return data_transformed


class SpikeWidth(BaseFeatureExtractor):
    """
    The angle at the right spike.
    """

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        left_index, right_index = get_percent_margin_point(data.iloc[:, 1:-1], percentage=0.5)
        left_index, right_index = left_index + 1, right_index + 1

        left_lines = get_lines_for_index(data, left_index)
        right_lines = get_lines_for_index(data, right_index)

        # calculate x intercept of the lines
        left_x_intercepts = - left_lines['k'] / left_lines['m']
        right_x_intercepts = - right_lines['k'] / right_lines['m']

        spike_width = right_x_intercepts - left_x_intercepts

        return spike_width

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'spike_width'
        return data_transformed


class NeoCoefficientMin(BaseFeatureExtractor):

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        data_reset_columns = data.copy()
        # Quick hack to get integer index

        data_reset_columns.columns = [i for i in range(len(data_reset_columns.columns))]
        min_index = data_reset_columns.iloc[:, 1:-1].idxmin(axis=1)
        neo = calculate_neo_coefficient(data_reset_columns, index=min_index)

        return neo.astype(float)

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'neo_coefficient_min'
        return data_transformed


class NeoCoefficientMax(BaseFeatureExtractor):

    @staticmethod
    def _calculate(data: pd.DataFrame) -> pd.Series:
        data_reset_columns = data.copy()
        # Quick hack to get integer index

        data_reset_columns.columns = [i for i in range(len(data_reset_columns.columns))]
        min_index = data_reset_columns.iloc[:, 1:-1].idxmax(axis=1)
        neo = calculate_neo_coefficient(data_reset_columns, index=min_index)

        return neo.astype(float)

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self._calculate(data)
        data_transformed.index = data.index
        data_transformed.name = 'neo_coefficient_max'
        return data_transformed


class Pca4Components(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.fit_model = None

    def fit(self, data: pd.DataFrame) -> None:
        pca = PCA(n_components=4)
        pca.fit(data)
        self.fit_model = pca

    def transform(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = data.copy()
        data_transformed = self.fit_model.transform(data)
        data_transformed = pd.DataFrame(
            data_transformed,
            index=data.index,
            columns=[f'pca_{i + 1}' for i in range(data_transformed.shape[1])]
        )
        return data_transformed


class FeatureExtractor:
    SupportedFeatures: Dict[str, Callable[[], BaseFeatureExtractor]] = {
        'positive_amplitude': PositiveAmplitude,
        'negative_amplitude': NegativeAmplitude,
        'positive_signal_energy': PositiveSignalEnergy,
        'negative_signal_energy': NegativeSignalEnergy,
        'left_spike_angle': LeftSpikeAngle,
        'right_spike_angle': RightSpikeAngle,
        'spike_width': SpikeWidth,
        'neo_coefficient_min': NeoCoefficientMin,
        'neo_coefficient_max': NeoCoefficientMax,
        'pca_4_components': Pca4Components,
    }

    def __init__(self, features: List[str]):
        self.features = features
        self._fitted_models = None

    def _setup_model(self, model_name: str) -> BaseFeatureExtractor:
        return self.SupportedFeatures[model_name]()

    def fit(self, data: pd.DataFrame):
        fit_models = []
        for feature in self.features:
            fit_on_data = data.copy()
            model = self._setup_model(feature)
            model.fit(fit_on_data)
            fit_models.append(model)
        self._fitted_models = fit_models

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out_data = []
        if len(self._fitted_models) == 0:
            raise ValueError("No fitted model found. Did you fit the model correctly?")
        for model in self._fitted_models:
            transformed_data = model.transform(data)
            out_data.append(transformed_data)
        return pd.concat(out_data, axis=1)
