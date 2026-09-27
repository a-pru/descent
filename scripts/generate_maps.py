"""Builds the directed lane maps and lane distance fields (SDFs) used by DESCENT.

The input are the Amelia-10 semantic graphs (graph_data_a10v01os/<airport>/semantic_graph.pkl).

For every airport it writes to --out-dir:
    <airport>_custom_map.pt: {'start_lane': {id: segment}, 'lane_tree': {id: [segments]}}
        A segment is {'start', 'end', 'successors', 'predecessors', 'type'} in km (Amelia x/y).
        The lane tree of a segment contains the segments reachable forward and backward within
        20 hops without turning by more than 45 degrees, plus all segments with an endpoint
        within 0.5 km of one of its endpoints.
    <airport>_sdf.pt: {'sdf_grid': (H, W), 'metadata': {...}}, distance (km) to the nearest
        segment on a 5 m grid; used by the lane-adherence loss.

usage: python scripts/generate_maps.py --airports kbos ksfo [--sdf-only] [--plot]
"""

import argparse
import math
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

AIRPORTS = ["kbos", "kdca", "kewr", "kjfk", "klax", "kmdw", "kmsy", "ksea", "ksfo", "panc"]


def get_vector(seg):
    """Direction vector of a segment."""
    return (seg["end"][0] - seg["start"][0], seg["end"][1] - seg["start"][1])


def get_angle(v1, v2):
    """Angle between two vectors in degrees (0 for zero-length vectors)."""
    mag1, mag2 = math.hypot(v1[0], v1[1]), math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    return math.degrees(math.acos(max(min(cos_theta, 1.0), -1.0)))


def get_close(start_id, lane_segments, x=0.5):
    """IDs of all segments with an endpoint within x of an endpoint of the start segment."""
    ref_s, ref_e = lane_segments[start_id]["start"], lane_segments[start_id]["end"]
    lanes = np.array([[s["start"], s["end"]] for s in lane_segments.values()])
    starts, ends = lanes[:, 0], lanes[:, 1]
    mask = (
        (np.linalg.norm(starts - ref_s, axis=-1) <= x)
        | (np.linalg.norm(ends - ref_e, axis=-1) <= x)
        | (np.linalg.norm(starts - ref_e, axis=-1) <= x)
        | (np.linalg.norm(ends - ref_s, axis=-1) <= x)
    )
    all_ids = np.array(list(lane_segments.keys()))
    return list(all_ids[np.where(mask)[0]])


def get_successors_constrained(
    start_id, lane_segments, max_depths, max_angle_deg=45.0, reverse=False
):
    """IDs of the segments reachable from start_id by depth-first search.

    Follows successors (predecessors if reverse) without exceeding the per-type depth or the turn
    angle.
    """
    collected_ids = set()

    def _recurse(current_id, depth):
        """Visits a segment and its admissible neighbours."""
        if current_id not in lane_segments:
            return
        current = lane_segments[current_id]
        if depth > max_depths.get(current.get("type"), max_depths.get("default", 10)):
            return
        collected_ids.add(current_id)

        next_ids = current.get("predecessors" if reverse else "successors", [])
        curr_vec = get_vector(current)
        for next_id in next_ids:
            if next_id in collected_ids or next_id not in lane_segments:
                continue
            next_vec = get_vector(lane_segments[next_id])
            angle = get_angle(next_vec, curr_vec) if reverse else get_angle(curr_vec, next_vec)
            if angle > max_angle_deg:
                continue
            _recurse(next_id, depth + 1)

    _recurse(start_id, 0)
    return list(collected_ids)


def simplify_graph_constrained(G, allowed_types=None, max_new_length=None):
    """Removes pass-through nodes and bridges them with a single edge.

    A node is removed if it has an allowed type, exactly two neighbours of the same type, and the
    new edge is at most max_new_length long.
    """

    def get_dist(n1, n2):
        """Euclidean distance between two graph nodes (0 if a position is missing)."""
        pos1, pos2 = G.nodes[n1], G.nodes[n2]
        if "x" not in pos1 or "y" not in pos1 or "x" not in pos2 or "y" not in pos2:
            return 0.0
        return math.hypot(pos1["x"] - pos2["x"], pos1["y"] - pos2["y"])

    if isinstance(allowed_types, (str, int)):
        allowed_types = [allowed_types]

    simplified = True
    while simplified:
        simplified = False
        nodes_to_remove = []
        for node in list(G.nodes()):
            if node in nodes_to_remove:
                continue
            node_data = G.nodes[node]
            if allowed_types is not None and node_data.get("node_type") not in allowed_types:
                continue

            neighbors = set(G.predecessors(node)).union(set(G.successors(node)))
            neighbors.discard(node)
            if len(neighbors) != 2:
                continue
            u, w = list(neighbors)
            if not (
                G.nodes[u].get("node_type")
                == node_data.get("node_type")
                == G.nodes[w].get("node_type")
            ):
                continue
            if max_new_length is not None and get_dist(u, w) > max_new_length:
                continue

            added_new_edge = False
            for a, b in [(u, w), (w, u)]:
                if G.has_edge(a, node) and G.has_edge(node, b):
                    ab_data, bb_data = G.get_edge_data(a, node)[0], G.get_edge_data(node, b)[0]
                    new_attr = ab_data.copy()
                    if "weight" in ab_data and "weight" in bb_data:
                        new_attr["weight"] = ab_data["weight"] + bb_data["weight"]
                    G.add_edge(a, b, **new_attr)
                    added_new_edge = True

            if added_new_edge:
                nodes_to_remove.append(node)
                simplified = True

        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
    return G


def build_lane_segments(graph_file):
    """Returns the directed lane segments (graph edges) with successor/predecessor IDs."""
    with open(graph_file, "rb") as f:
        semantic_map = pickle.load(f)
    graph = semantic_map["graph_networkx"]

    # Snap the graph nodes (lat/lon) to the closest polyline point and use its x/y.
    lut = {
        (line[0], line[1]): (line[2], line[3])
        for line in semantic_map["map_infos"]["all_polylines"]
    }
    points_ll = np.array(list(lut.keys()))
    points_xy = np.array(list(lut.values()))
    for _, data in graph.nodes(data=True):
        closest_idx = np.argmin(
            np.linalg.norm(points_ll - np.array([data["y"], data["x"]]), axis=1)
        )
        new_x, new_y = points_xy[closest_idx]
        data["x"], data["y"] = new_y, new_x

    graph = simplify_graph_constrained(graph, [3], 0.8)
    graph = simplify_graph_constrained(graph, [4], 0.03)
    graph = nx.relabel_nodes(
        graph, {n: "node_{:04d}".format(i) for i, n in enumerate(graph.nodes())}
    )

    lane_segments = {}
    for u, v in graph.edges():
        t1, t2 = graph.nodes[u]["node_type"], graph.nodes[v]["node_type"]
        successors = [f"{e[0]}_{e[1]}" for e in graph.out_edges([v]) if e[1] != u]
        predecessors = [f"{e[0]}_{e[1]}" for e in graph.in_edges([u]) if e[1] != v]
        lane_segments[f"{u}_{v}"] = {
            "start": [graph.nodes[u]["x"], graph.nodes[u]["y"]],
            "end": [graph.nodes[v]["x"], graph.nodes[v]["y"]],
            "successors": successors,
            "predecessors": predecessors,
            "type": t1 if t1 == t2 else -1,
        }
    return lane_segments


def build_custom_map(lane_segments):
    """Lane tree (PRS candidates) of every segment."""
    custom_map = {"lane_tree": {}, "start_lane": {}}
    max_depths = {-1: 20, 3: 20, 4: 20, 5: 20}
    for start_id in tqdm(list(lane_segments.keys())):
        succ_ids = get_successors_constrained(start_id, lane_segments, max_depths=max_depths)
        prev_ids = get_successors_constrained(
            start_id, lane_segments, max_depths=max_depths, reverse=True
        )
        close_ids = get_close(start_id, lane_segments)
        ids = sorted(set(succ_ids + prev_ids + close_ids))
        custom_map["lane_tree"][start_id] = [lane_segments[i] for i in ids]
        custom_map["start_lane"][start_id] = lane_segments[start_id]
    return custom_map


def precompute_airport_sdf(lane_segments, resolution, padding=0.0):
    """Unsigned distance (km) from every grid cell to the nearest lane segment."""
    segments = [
        (
            torch.tensor(seg["start"], dtype=torch.float32),
            torch.tensor(seg["end"], dtype=torch.float32),
        )
        for seg in lane_segments.values()
    ]
    all_coords = torch.stack([p for seg in segments for p in seg])
    min_coords = all_coords.min(dim=0)[0] - padding
    max_coords = all_coords.max(dim=0)[0] + padding

    width = int((max_coords[0] - min_coords[0]) / resolution)
    height = int((max_coords[1] - min_coords[1]) / resolution)
    grid_x = torch.linspace(min_coords[0], max_coords[0], width)
    grid_y = torch.linspace(min_coords[1], max_coords[1], height)
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid_points = torch.stack([mesh_x, mesh_y], dim=-1).view(-1, 2)

    sdf_grid = torch.full((grid_points.shape[0],), float("inf"))
    for p1, p2 in segments:
        ab, ap = p2 - p1, grid_points - p1
        t = torch.clamp(torch.sum(ap * ab, dim=-1) / torch.sum(ab * ab), 0, 1)
        if torch.isnan(t).any():  # Zero-length segment
            continue
        dist = torch.norm(grid_points - (p1 + t.unsqueeze(-1) * ab), dim=-1)
        sdf_grid = torch.min(sdf_grid, dist)

    metadata = {
        "min_coords": min_coords.tolist(),
        "max_coords": max_coords.tolist(),
        "resolution": resolution,
        "shape": (height, width),
    }
    return sdf_grid.view(height, width), metadata


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--airports", nargs="+", default=AIRPORTS)
    parser.add_argument("--graph-dir", default="datasets/amelia/graph_data_a10v01os")
    parser.add_argument("--out-dir", default="maps")
    parser.add_argument("--sdf-only", action="store_true", help="only rebuild the SDFs")
    parser.add_argument("--plot", action="store_true", help="also save a PNG of each SDF")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for airport in args.airports:
        print(f"Processing {airport}")
        lane_segments = build_lane_segments(
            os.path.join(args.graph_dir, airport, "semantic_graph.pkl")
        )

        if not args.sdf_only:
            custom_map = build_custom_map(lane_segments)
            torch.save(custom_map, os.path.join(args.out_dir, f"{airport}_custom_map.pt"))

        sdf_grid, metadata = precompute_airport_sdf(lane_segments, resolution=5 / 1000.0)
        torch.save(
            {"sdf_grid": sdf_grid, "metadata": metadata},
            os.path.join(args.out_dir, f"{airport}_sdf.pt"),
        )

        if args.plot:
            plt.figure(figsize=(20, 20))
            plt.imshow(sdf_grid.numpy(), cmap="hot", origin="lower")
            plt.colorbar(label="Distance to nearest lane segment (km)")
            plt.savefig(os.path.join(args.out_dir, f"{airport}_sdf.png"))
            plt.close()


if __name__ == "__main__":
    main()
