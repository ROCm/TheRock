# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json


class ReduceConfig:
    def __init__(self, config_json_path: str):
        self.config_json_path = config_json_path
        with open(config_json_path, "r") as f:
            self.config_dict = json.load(f)
