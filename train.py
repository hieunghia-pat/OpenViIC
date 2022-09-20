import argparse

from builders.trainer_builder import build_trainer
from configs.utils import get_config

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)

args = parser.parse_args()

config = get_config(args.config_file)

trainer = build_trainer(config)

trainer.start()