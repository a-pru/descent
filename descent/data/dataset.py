"""Amelia-10 scenes in the ego frame with PRS lane context."""

import os
import pickle
import random
from math import cos, radians, sin
from typing import Dict

import numpy as np
import torch
from easydict import EasyDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import descent.utils.data_utils as D
import descent.utils.global_masks as G
from descent.utils import pylogger

log = pylogger.get_pylogger(__name__)

# Use single-threaded MKL in the loader workers. File-system sharing avoids file handle limits.
os.environ["MKL_NUM_THREADS"] = "1"
torch.multiprocessing.set_sharing_strategy("file_system")


class DescentDataset(Dataset):
    """Amelia-10 scenes in the ego-agent frame with the ego agent's PRS lane segments.

    The PRS (Potential Reachable Set) is the precomputed lane tree of the ego agent's current
    segment.
    """

    def __init__(self, config: EasyDict) -> None:
        """Reads the dataset parameters; the data itself is loaded in prepare_data.

        Args:
            config: Dataset parameters (see configs/data/default.yaml).
        """
        super().__init__()
        self.in_data_dir = config.in_data_dir
        self.map_dir = config.map_dir

        self.hist_len = config.hist_len
        self.pred_lens = config.pred_lens
        self.pred_len = max(self.pred_lens)
        self.curr_timestep = self.hist_len - 1

        self.min_agents = config.min_agents
        self.max_agents = config.max_agents

        self.sampling_strategy = config.sampling_strategy
        self.supported_strategies = config.supported_sampling_strategies
        assert self.sampling_strategy in self.supported_strategies, (
            f"Strategy {self.sampling_strategy} not in supported list {self.supported_strategies}"
        )
        self.k_agents = config.k_agents

        self.seed = config.seed
        # If False, the ego agent is the first agent of the sampling order ('critical' agent).
        # If True, it is drawn uniformly from the k agents ('random' agent).
        self.random_ego_agent = config.random_ego_agent
        # Only every n-th train/val scenario is used (n = 2 for the paper models). Test uses all.
        self.scenario_subsample = config.scenario_subsample

    def set_split_list(self, split_path: str) -> None:
        """Reads the list of hour shards of this split."""
        with open(split_path, "r") as fp:
            self.split_list = [line.strip() for line in fp]

    def prepare_data(self, split: str) -> None:
        """Loads the lane maps and SDFs of the split's airports and lists its scene files."""
        log.info(f"Preparing {split} data.")
        self.test = split == "test"
        self.custom_maps, self.sdfs, self.scenario_list = {}, {}, {}

        airports = list(set([f.split("/")[0] for f in self.split_list]))
        for airport in airports:
            self.scenario_list[airport] = D.get_filtered_list(
                airport, self.in_data_dir, self.split_list, self.min_agents, self.max_agents
            )

            # Directed lane map from scripts/generate_maps.py. 'start_lane' holds all segments and
            # 'lane_tree' the segments reachable from each of them.
            input_map = torch.load(
                os.path.join(self.map_dir, f"{airport}_custom_map.pt"), weights_only=False
            )
            self.custom_maps[airport] = {
                "lane_trees": list(input_map["lane_tree"].values()),
                "root_lane_points": np.stack(
                    [
                        np.linspace(lane["start"], lane["end"], num=10, endpoint=True)[:, [1, 0]]
                        for lane in input_map["start_lane"].values()
                    ],
                    axis=0,
                ),
                "root_lane_types": [lane["type"] for lane in input_map["start_lane"].values()],
            }
            self.sdfs[airport] = torch.load(
                os.path.join(self.map_dir, f"{airport}_sdf.pt"), weights_only=False
            )

        # Use the same number of scenarios per airport, drawn with a fixed seed. This also
        # shuffles the scenario order.
        files_per_airport = min(len(f) for _, f in self.scenario_list.items())
        balanced_list = []
        for airport, files in self.scenario_list.items():
            random.seed(self.seed)
            random.shuffle(files)
            balanced_list += files[:files_per_airport]
        self.scenario_list = balanced_list

        if not self.test:
            self.scenario_list = self.scenario_list[:: self.scenario_subsample]
        log.info(f"{split}: {len(self.scenario_list)} scenarios from {sorted(airports)}")

    def transform_custom_map(self, custom_map: Dict, sequences: np.ndarray, ego_agent: int):
        """Selects the ego agent's PRS lane segments and moves them to its frame.

        The ego agent is matched to the closest of the 5 nearest lane segments whose direction is
        within 90 degrees of its heading (fallback: the nearest segment). The lane tree of that
        segment is the map context. If the agent stood still over the whole history (speed < 1),
        or is slower than 15 next to a segment of type 3 or an unknown type (< 0), the frame
        heading is the heading of the last checked segment instead of the reported heading.

        Returns:
            local_context: M lane segments (M, 20, 2), 20 points each, in the ego frame.
            lane_types: Lane type + 1 (M,).
            ego_position: Global ego position (3,), the frame origin.
            ego_heading: Frame heading in radians.
        """
        ego_position = sequences[ego_agent, self.curr_timestep, G.XYZ]
        ego_position = torch.from_numpy(ego_position).float()
        ego_position_xy = ego_position[..., :2].cpu().numpy()
        ego_heading = radians(sequences[ego_agent, self.curr_timestep, G.SEQ_IDX.Heading])

        root_lane_points = custom_map["root_lane_points"]
        lane_sets = custom_map["lane_trees"]

        dists = np.linalg.norm(root_lane_points - ego_position_xy, axis=-1)
        sorted_segment_indices = np.argsort(dists.min(axis=-1))

        matched = False
        for i in range(5):
            closest_segment_idx = sorted_segment_indices[i]
            lane_vec = (
                root_lane_points[closest_segment_idx, -1] - root_lane_points[closest_segment_idx, 0]
            )
            lane_heading = np.arctan2(lane_vec[1], lane_vec[0])
            ang_diff = (ego_heading - lane_heading + np.pi) % (2 * np.pi) - np.pi
            if abs(np.degrees(ang_diff)) < 90:
                matched = True
                break
        if not matched:
            closest_segment_idx = sorted_segment_indices[0]

        type_of_closest = custom_map["root_lane_types"][closest_segment_idx]
        speed = sequences[ego_agent, : self.curr_timestep + 1, G.SEQ_IDX.Speed]
        if (speed < 1.0).all() or (
            speed[-1] < 15.0 and (type_of_closest == 3 or type_of_closest < 0)
        ):
            ego_heading = lane_heading
        selected_lanes = lane_sets[closest_segment_idx]

        sampling_steps = 20
        points, lane_types = [], []
        for raw_lane in selected_lanes:
            lane = np.linspace(
                raw_lane["start"], raw_lane["end"], num=sampling_steps, endpoint=True
            )
            points.append(lane[:, [1, 0]])
            lane_types.append(raw_lane["type"] + 1)
        global_context = np.stack(points, axis=0)
        lane_types = torch.tensor(lane_types)

        R = np.array(
            [
                [np.cos(ego_heading), -np.sin(ego_heading)],
                [np.sin(ego_heading), np.cos(ego_heading)],
            ]
        )
        local_context = (global_context - ego_position_xy) @ R
        local_context = torch.from_numpy(local_context.reshape(-1, sampling_steps, 2))
        return local_context, lane_types, ego_position, ego_heading

    def transform_sequences(
        self, sequences: np.ndarray, ego_position: torch.Tensor, ego_heading: float
    ) -> np.ndarray:
        """Moves all agents to the ego frame.

        Returns:
            Sequences (A, T, 5) with x, y, z (km), speed and heading (rad).
        """
        num_agents, timesteps, _ = sequences.shape
        rel_sequence = np.zeros(shape=(num_agents, timesteps, 5))

        R = np.array(
            [
                [cos(ego_heading), -sin(ego_heading), 0.0],
                [sin(ego_heading), cos(ego_heading), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R = np.repeat(R.reshape(1, 3, 3), num_agents, axis=0)

        rel_xyz = sequences[:, :, G.XYZ] - ego_position.numpy()
        rel_sequence[:, :, :3] = np.matmul(rel_xyz, R)
        rel_sequence[:, :, -1] = D.wrap_angle(
            sequences[:, :, G.SEQ_IDX.Heading] - np.degrees(ego_heading)
        )
        rel_sequence[:, :, -2] = sequences[:, :, G.SEQ_IDX.Speed]
        return rel_sequence

    def transform_scene_data(self, scene_data: Dict) -> Dict:
        """Selects the k agents and the ego agent and transforms the scene to the ego frame.

        The ego agent is moved to index 0.
        """
        sequences = scene_data["agent_sequences"]
        agent_masks = scene_data["agent_masks"]
        airport_id = scene_data["airport_id"]

        agents_in_scene = scene_data["meta"]["agent_order"][self.sampling_strategy][: self.k_agents]
        num_agents = len(agents_in_scene)
        if self.random_ego_agent:
            if self.test:
                random.seed(scene_data["scenario_id"])  # Same ego agent on every test run
            ego_agent = random.randint(a=0, b=num_agents - 1)
        else:
            ego_agent = 0

        sequences = sequences[agents_in_scene]
        agent_masks = agent_masks[agents_in_scene]

        i = ego_agent
        sequences = np.concatenate(
            (sequences[i : i + 1], sequences[:i], sequences[i + 1 :]), axis=0
        )
        agent_masks = np.concatenate(
            (agent_masks[i : i + 1], agent_masks[:i], agent_masks[i + 1 :]), axis=0
        )
        ego_agent = 0

        local_context, lane_types, ego_position, ego_heading = self.transform_custom_map(
            self.custom_maps.get(airport_id), sequences, ego_agent
        )
        rel_sequences = self.transform_sequences(sequences, ego_position, ego_heading)

        agent_types = torch.tensor(scene_data["agent_types"])
        agent_types = agent_types[agents_in_scene].view(-1)

        return {
            "scenario_id": scene_data["scenario_id"],
            "airport_id": airport_id,
            "agent_types": agent_types,
            "agent_masks": agent_masks,
            "ego_agent_id": ego_agent,
            "num_agents": sequences.shape[0],
            "rel_sequences": rel_sequences,
            "custom_context": local_context,
            "lane_padding_mask": torch.zeros_like(local_context[..., -1], dtype=torch.bool),
            "x_padding_mask": torch.from_numpy((~agent_masks)).float(),
            "lane_types": lane_types,
            "origin": ego_position.view(3),
            "theta": torch.tensor(ego_heading).view(1),
            "sdf": self.sdfs.get(airport_id),
        }

    def collate_batch(self, batch_data: Dict) -> Dict:
        """Stacks the scenes of a batch and pads agents and lane segments."""
        batch_size = len(batch_data)
        key_to_list = {key: [b[key] for b in batch_data] for key in batch_data[0].keys()}

        input_dict = {}
        for key, val_list in key_to_list.items():
            if key in ["scenario_id", "airport_id", "ego_agent_id", "num_agents"]:
                input_dict[key] = np.asarray(val_list)
            elif key == "rel_sequences":
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = D.merge_seq3d_by_padding(val_list, max_pad=self.k_agents)
            elif key == "agent_masks":
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = D.merge_seq2d_by_padding(val_list, max_pad=self.k_agents)
            elif key in ["custom_context", "lane_types", "agent_types", "origin", "theta"]:
                input_dict[key] = pad_sequence(val_list, batch_first=True)
            elif key in ["x_padding_mask", "lane_padding_mask"]:
                input_dict[key] = pad_sequence(val_list, batch_first=True, padding_value=True)
            elif key == "sdf":
                # The batch uses the SDF of its first scene. This is exact for single-airport
                # training. In multi-airport batches, scenes from other airports are scored on
                # this airport's SDF, as for the released seen-all model.
                input_dict[key] = val_list[0]
            else:
                raise KeyError(f"No collate rule for '{key}'")
        input_dict["x_key_padding_mask"] = input_dict["x_padding_mask"].all(-1)
        input_dict["lane_key_padding_mask"] = input_dict["lane_padding_mask"].all(-1)
        return {"batch_size": batch_size, "scene_dict": input_dict}

    def __len__(self):
        return len(self.scenario_list)

    def __getitem__(self, index):
        """Loads a scene file and transforms it to the ego frame."""
        with open(self.scenario_list[index], "rb") as f:
            data = pickle.load(f)
        return self.transform_scene_data(data)
