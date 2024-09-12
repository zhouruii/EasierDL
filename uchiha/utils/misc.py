import os.path
from typing import Union


def strings_to_list(x: Union[str, dict]):
    if isinstance(x, dict):
        for key, value in x.items():
            if isinstance(value, str):
                splits = value.split(',')
            else:
                continue
            if len(splits) > 1:
                new_value = [int(num) if num.strip().isdigit() else num.strip() for num in splits]
                x[key] = new_value
    else:
        splits = x.split(',')
        return [int(num) if num.strip().isdigit() else num.strip() for num in splits]
    return x


def get_extension(path):
    basename = os.path.basename(path)
    return basename.split('.')[-1]
