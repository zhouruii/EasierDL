from typing import Union


def strings_to_list(x: Union[str, dict]):
    if isinstance(x, dict):
        for key, value in x.items():
            if isinstance(value, str):
                splits = value.split(',')
            else:
                continue
            if len(splits) > 1:
                new_value = [int(num) for num in splits]
                x[key] = new_value
    else:
        splits = x.split(',')
        if len(splits) > 1:
            return [int(num) for num in splits]
    return x

