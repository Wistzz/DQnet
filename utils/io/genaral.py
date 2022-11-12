# -*- coding: utf-8 -*-

import contextlib
import json
import os
from collections import defaultdict


def read_data_from_json(json_path: str) -> dict:
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_data_from_txt(path: str) -> list:
    """
    读取文件中各行数据，存放到列表中
    """
    lines = []
    with open(path, encoding="utf-8", mode="r") as f:
        line = f.readline().strip()
        while line:
            lines.append(line)
            line = f.readline().strip()
    return lines


def get_name_list_from_dir(path: str) -> list:

    return [os.path.splitext(x)[0] for x in os.listdir(path)]


def get_datasets_info_with_keys(dataset_infos: list, extra_keys: list) -> dict:


    # total_keys = tuple(set(extra_keys + ["image"]))
    # e.g. ('image', 'mask')
    def _get_intersection(list_a: list, list_b: list, to_sort: bool = True):

        intersection_list = list(set(list_a).intersection(set(list_b)))
        if to_sort:
            return sorted(intersection_list)
        return intersection_list

    def _get_info(dataset_info: dict, extra_keys: list, path_collection: defaultdict):

        total_keys = tuple(set(extra_keys + ["image"]))
        # e.g. ('image', 'mask')

        dataset_root = dataset_info.get("root", ".")

        infos = {}
        for k in total_keys:
            assert k in dataset_info, f"{k} is not in {dataset_info}"
            infos[k] = dict(dir=os.path.join(dataset_root, dataset_info[k]["path"]), ext=dataset_info[k]["suffix"])

        if (index_file_path := dataset_info.get("index_file", None)) is not None:
            image_names = get_data_from_txt(index_file_path)
        else:
            image_names = get_name_list_from_dir(infos["image"]["dir"])

        if "mask" in total_keys:
            mask_names = get_name_list_from_dir(infos["mask"]["dir"])
            image_names = _get_intersection(image_names, mask_names)

        for i, name in enumerate(image_names):
            for k in total_keys:
                path_collection[k].append(os.path.join(infos[k]["dir"], name + infos[k]["ext"]))

    path_collection = defaultdict(list)
    for dataset_name, dataset_info in dataset_infos:
        prev_num = len(path_collection["image"])
        _get_info(dataset_info=dataset_info, extra_keys=extra_keys, path_collection=path_collection)
        curr_num = len(path_collection["image"])
        print(f"Loading data from {dataset_name}: {dataset_info['root']} ({curr_num - prev_num})")
    return path_collection


@contextlib.contextmanager
def open_w_msg(file_path, mode, encoding=None):

    
    f = open(file_path, encoding=encoding, mode=mode)
    yield f
    
    f.close()
