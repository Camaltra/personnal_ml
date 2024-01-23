import os.path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob
import numpy as np
import open3d as o3d


class S3DISDataset(Dataset):
    def __init__(
        self,
        root: str,
        areas_nums: str,
        split: str = "train",
        num_points: int = 2048,
        r_prob=0.25,
    ):
        super().__init__()
        self.split = split
        self.areas_nums = areas_nums
        self.root = root
        self.num_points = num_points
        self.r_prob = r_prob

        if not os.path.exists(root):
            raise Exception

        areas = glob(os.path.join(root, f"Area_[{areas_nums}]"))
        if not areas:
            raise FileNotFoundError

        for area in areas:
            if not os.path.exists(area):
                raise FileNotFoundError

        self.data_paths = []
        for area in areas:
            self.data_paths += glob(os.path.join(area, "*.hdf5"), recursive=True)

    def __getitem__(self, ix):
        path = self.data_paths[ix]
        xyzc = pd.read_hdf(path, key="space_slice").to_numpy()

        xyz = xyzc[:, :3]
        c = xyzc[:, 3]
        if self.num_points:
            xyz, c = self.sample_fps(xyz, c)

        if self.split != "test":
            xyz += np.random.normal(0.0, 0.02, xyz.shape)

            if np.random.uniform(0, 1) > 1 - self.r_prob:
                xyz = self.random_rotate(xyz)

        xyz = self.normalize_point(xyz)

        x = torch.from_numpy(xyz).type(torch.float32)
        targets = torch.from_numpy(c).type(torch.LongTensor)

        return x, targets

    @staticmethod
    def random_rotate(points):
        psi = np.random.uniform(-np.pi, np.pi)

        rot_z = np.array(
            [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
        )

        return np.matmul(points, rot_z)

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def normalize_point(points):
        points = points - points.min(axis=0)
        points /= points.max(axis=0)

        return points

    def sample_fps(self, points, target):
        N = points.shape[0]
        centroids = np.zeros(self.num_points, dtype=int)
        distance = np.full(N, np.inf)
        farthest = np.random.randint(0, N)

        for i in range(self.num_points):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, axis=0)

        sampled_points = points[centroids]
        sampled_targets = target[centroids]

        return sampled_points, sampled_targets


COLOR_MAP = {
    0: (47, 79, 79),  # ceiling - darkslategray
    1: (139, 69, 19),  # floor - saddlebrown
    2: (34, 139, 34),  # wall - forestgreen
    3: (75, 0, 130),  # beam - indigo
    4: (255, 0, 0),  # column - red
    5: (255, 255, 0),  # window - yellow
    6: (0, 255, 0),  # door - lime
    7: (0, 255, 255),  # table - aqua
    8: (0, 0, 255),  # chair - blue
    9: (255, 0, 255),  # sofa - fuchsia
    10: (238, 232, 170),  # bookcase - palegoldenrod
    11: (100, 149, 237),  # board - cornflower
    12: (255, 105, 180),  # stairs - hotpink
    13: (0, 0, 0),  # clutter - black
}


if __name__ == "__main__":
    map_colors = lambda x: COLOR_MAP[x]
    v_map_colors = np.vectorize(map_colors)
    root = "./data/Stanford3dDataset_v1.2_Reduced_Partitioned_Aligned_Version/"
    a = S3DISDataset(root=root, areas_nums="1")
    dl = DataLoader(a, shuffle=True, batch_size=1)

    for b, c in dl:
        b = b.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        b = b.squeeze()
        c = c.squeeze()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(b)
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(v_map_colors(c)).T / 255)

        o3d.visualization.draw_geometries(pcd)
        break
