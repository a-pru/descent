"""Scene file listing, blacklist handling and padding helpers."""

import os
from typing import List

import numpy as np
import torch
from easydict import EasyDict


def wrap_angle(angle: np.array) -> np.array:
    """Wraps angles in degrees to [-180, 180) and converts them to radians."""
    return np.radians(((angle % 360) + 540) % 360 - 180)


def get_filtered_list(
    airport: str, base_dir: str, file_list: List[str], min_agents: int, max_agents: int
) -> List[str]:
    """Lists the scene files of an airport with min_agents to max_agents agents.

    Scene files are named '<scene_idx>_n-<num_agents>.pkl'.
    """
    data_dirs = [os.path.join(base_dir, f) for f in file_list if airport in f]

    data_files = []
    for dir_ in data_dirs:
        if os.path.exists(dir_):
            data_files.extend([os.path.join(dir_, f) for f in os.listdir(dir_)])
        else:
            print(f"\033[93mWarning:\033[0m Directory {dir_} does not exist. Skipping.")

    def num_agents(f):
        """Number of agents encoded in the scene file name."""
        return int(f.split("/")[-1].split(".")[0].split("-")[-1])

    return [f for f in data_files if min_agents <= num_agents(f) <= max_agents]


def load_blacklist(data_prep: EasyDict, airport_list: list) -> dict:
    """Reads the blacklisted scene directories of each airport (empty if there is no blacklist)."""
    blacklist_dir = os.path.join(data_prep.in_data_dir, "blacklist")
    blacklist = {}
    for airport in airport_list:
        blacklist_file = os.path.join(blacklist_dir, f"{airport}_{data_prep.split_type}.txt")
        blacklist[airport] = []
        if os.path.exists(blacklist_file):
            with open(blacklist_file, "r") as f:
                blacklist[airport] = f.read().splitlines()
    return blacklist


def flatten_blacklist(blacklist: dict) -> list:
    """Merges the per-airport blacklists into one list."""
    return [f for files in blacklist.values() for f in files]


def remove_blacklisted(blacklist: list, file_list: list) -> list:
    """Removes blacklisted entries from file_list (in place) and returns it."""
    for duplicate in list(set(blacklist) & set(file_list)):
        file_list.remove(duplicate)
    return file_list


def merge_seq3d_by_padding(tensor_list: List[torch.tensor], max_pad: int = None) -> torch.tensor:
    """Stacks (N_i, T, D) tensors into (B, max_pad, T, D), zero-padding the agent dimension."""
    assert len(tensor_list[0].shape) == 3
    max_num_agents = max([x.shape[0] for x in tensor_list]) if max_pad is None else max_pad
    _, timesteps, dims = tensor_list[0].shape

    ret_tensor_list = []
    for cur_tensor in tensor_list:
        assert cur_tensor.shape[2] == dims
        new_tensor = cur_tensor.new_zeros(max_num_agents, timesteps, dims)
        new_tensor[: cur_tensor.shape[0], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)
    return torch.stack(ret_tensor_list, dim=0)


def merge_seq2d_by_padding(tensor_list: List[torch.tensor], max_pad: int = None) -> torch.tensor:
    """Stacks (N_i, T) tensors into (B, max_pad, T), zero-padding the agent dimension."""
    assert len(tensor_list[0].shape) == 2
    max_num_agents = max([x.shape[0] for x in tensor_list]) if max_pad is None else max_pad
    _, timesteps = tensor_list[0].shape

    ret_tensor_list = []
    for cur_tensor in tensor_list:
        assert cur_tensor.shape[-1] == timesteps
        new_tensor = cur_tensor.new_zeros(max_num_agents, timesteps)
        new_tensor[: cur_tensor.shape[0]] = cur_tensor
        ret_tensor_list.append(new_tensor)
    return torch.stack(ret_tensor_list, dim=0)
