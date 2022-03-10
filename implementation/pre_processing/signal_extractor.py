import logging
from dataclasses import dataclass
from statistics import median
from typing import NamedTuple, Optional
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import io
from scipy import signal


# ---------- CONFIGURATION ----------

class SignalExtractingConfiguration(NamedTuple):
    """
    peak_distance: the area in which only one peak can occur
    threshold_factor:  the threshold multiplier to use
    lower_boundary_signal: the lower bound (e.g. -150)
    upper_boundary_signal: the lower bound (e.g. 150)
    """
    peak_distance: int
    threshold_factor: float
    lower_boundary_signal: int
    upper_boundary_signal: int


# --------- PRIVATE FUNCTIONS ---------

def _extract_one_peak(signal_data: pd.Series,
                      index_peak: int,
                      lower_boundary: float,
                      upper_boundary: float,
                      ) -> pd.DataFrame:
    """
    Extracts one peak from a given index
    """
    start_index = index_peak + lower_boundary
    end_index = index_peak + upper_boundary + 1
    peak_data = signal_data.iloc[start_index:end_index]
    return peak_data


def _find_closest_peak(min_max: Tuple[int, int], peak: int, labels: pd.Series) -> int:
    diffs = [abs(label - peak) for label in labels.index.tolist()]
    min_index_of_index = int(np.argmin(diffs))
    if min_max[0] <= diffs[min_index_of_index] <= min_max[1]:
        return labels.iloc[min_index_of_index]
    else:
        return 0


# --------- PUBLIC FUNCTIONS ---------


def calculate_threshold(signal_data: pd.Series, multiplier: float = 5) -> float:
    threshold_data_median = median(signal_data.abs() / 0.6745)
    return multiplier * threshold_data_median


def create_peak_matrix(data: pd.Series, config: SignalExtractingConfiguration) -> pd.DataFrame:
    """
    Creates a DataFrame containing all the peaks. Rows index is the number of peak, column index the data point
    """
    cols_peak = [i for i in range(config.lower_boundary_signal, config.upper_boundary_signal + 1)]
    threshold = calculate_threshold(data, config.threshold_factor)
    peak_indexes = signal.find_peaks(data.values, threshold, distance=config.peak_distance)[0]

    # Create the DataFrame
    all_peaks_list = []
    for peak_index in peak_indexes:
        one_peak_data = _extract_one_peak(data, peak_index, config.lower_boundary_signal, config.upper_boundary_signal)
        one_peak_dataframe = pd.DataFrame(data=[one_peak_data.values.tolist()],
                                          columns=cols_peak, index=[peak_index])
        all_peaks_list.append(one_peak_dataframe)
    all_peaks_dataframe = pd.concat(all_peaks_list)
    return all_peaks_dataframe


def assign_labels_to_identified_peaks(labels: pd.Series,
                                      peaks: pd.DataFrame,
                                      config: SignalExtractingConfiguration
                                      ) -> pd.Series:
    labels = labels.loc[labels != 0]
    all_list = {}
    for peak in peaks.index.tolist():
        all_list[peak] = _find_closest_peak((-config.peak_distance, config.peak_distance), peak, labels)
    return pd.DataFrame(all_list.values(), index=all_list.keys())[0]
