import os
import numpy as np
import yaml
import csv
from datetime import datetime


def read_cfg(config_file):
    with open(config_file, 'r') as rf:
        data = yaml.safe_load(rf)
        return data


