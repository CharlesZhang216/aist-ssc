import pandas as pd
from config_xgb import (
    GROUP_NAMES,
    FEATS,
    LABEL,
    CATEGORICAL_FEATS,
    FEATS_TO_ENCODE,
    DATE_FEATS,
    TARGET_METRICS,
    FEATS_TO_DISCARD,
)
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics._scorer import make_scorer
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from scipy.stats import randint

from utils import SpaceTimeSplits, preprocess_feats
from metrics import get_metrics

pd.options.mode.chained_assignment = None

TO_NORM_FEATS = True # set to True to normalize features
TO_NORM_LABELS = False # set to True to normalize labels
TO_REMOVE_INC_COLS = True
TO_REMOVE_INC_ROWS = False
IS_PRED_EXP = False  # Set to True to directly predict TSS as np.exp(value). otherwise, estimate the ln of the actual TSS value and just modify for measuring metrics
