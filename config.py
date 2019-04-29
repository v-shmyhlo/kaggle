from ruamel.yaml import YAML


class Config(object):
    def __init__(self, config):
        if not isinstance(config, dict):
            raise ValueError('config must be dict')

        self.config = config

    def __getattr__(self, key):
        if key not in self.config:
            raise KeyError(key)

        value = self.config[key]

        if isinstance(value, dict):
            value = Config(value)

        return value

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls(YAML().load(f))
