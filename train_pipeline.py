import logging
import sys
from entities.train_pipeline_params import TrainingPipelineParams
from data.make_dataset import read_data, split_train_val_data
from features.build_features import extract_target, make_features
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_train_pipeline(params: TrainingPipelineParams):
    logger.info(f"Start training with params: {params}")
    data = read_data(params.input_data_path)
    train_set, val_set = split_train_val_data(data, params.splitting_params)
    y_train = extract_target(train_set, params)
