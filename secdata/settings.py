import os
import json


class Settings:
    def __init__(self, ini_file_path):
        self._config = json.load(open(ini_file_path, "r"))

    def __getitem__(self, key):
        return self._config[key]


settings = Settings(os.path.join(os.path.dirname(os.path.realpath(__file__)), "default_settings.json"))