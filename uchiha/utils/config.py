import yaml


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = Config(**value)
            setattr(self, key, value)

    def __repr__(self):
        return self._repr_recursive(self, 0)

    def _repr_recursive(self, d, indent_level):
        """递归地格式化字典为字符串"""
        indent = ' ' * (indent_level * 2)
        repr_str = ''
        for key, value in d.__dict__.items():
            if isinstance(value, Config):
                repr_str += f"{indent}{key}:\n"
                repr_str += self._repr_recursive(value, indent_level + 1)
            else:
                repr_str += f"{indent}{key}: {value}\n"
        return repr_str

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.to_dict()
            result[key] = value
        return result


def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


if __name__ == '__main__':
    yaml_path = 'configs/s_e512_wo-w.yaml'
    config = load_config(yaml_path)
    print(config)
    print(config.data.dataloader.to_dict())