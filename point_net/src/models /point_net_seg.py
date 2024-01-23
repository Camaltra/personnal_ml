import torch
import torch.nn as nn

import numpy as np
from torch.autograd import Variable


class TNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.shared_mlp = SharedMLP(
            self.input_dim, [64, 128, 1024], use_batch_norm=True, use_relu=True
        )
        self.fc_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim * self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        idd = (
            Variable(
                torch.from_numpy(np.eye(self.input_dim).flatten().astype(np.float32))
            )
            .view(1, self.input_dim * self.input_dim)
            .repeat(batch_size, 1)
            .to(x.device)
        )

        x = self.shared_mlp(x)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = self.fc_net(x) + idd
        return x.view(-1, self.input_dim, self.input_dim)


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        *,
        use_batch_norm: bool,
        use_relu: bool
    ) -> None:
        super().__init__()
        in_c = in_channels
        layers_c = []
        for out_c in out_channels:
            layers_c.append((in_c, out_c))
            in_c = out_c

        self.network = nn.ModuleList([])

        for in_c, out_c in layers_c:
            self.network.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(in_c, out_c, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm1d(out_c) if use_batch_norm else nn.Identity(),
                        nn.ReLU() if use_relu else nn.Identity(),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, norm, activation in self.network:
            x = activation(norm(conv(x)))
        return x


class PointNetBase(nn.Module):
    def __init__(self, input_dim: int = 3) -> None:
        super().__init__()
        self.input_transform = TNet(input_dim)
        self.shared_layer_one = SharedMLP(3, [64], use_batch_norm=True, use_relu=True)
        self.feature_transform = TNet(64)
        self.shared_layer_two = SharedMLP(
            64, [128, 1024], use_batch_norm=True, use_relu=True
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, features, nums_points = x.shape

        out_input_trms = self.input_transform(x)

        x = x.transpose(2, 1)
        if features > 3:
            other_features = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, out_input_trms)
        if features > 3:
            x = torch.concat(x, other_features, dim=2)
        x = x.transpose(2, 1)

        x = self.shared_layer_one(x)

        out_feature_trms = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, out_feature_trms)
        x = x.transpose(2, 1)

        x = self.shared_layer_two(x)

        out_max_pool = torch.max(x, 2, keepdim=True)[0]

        return out_input_trms, out_feature_trms, out_max_pool


class PointNetSeg(nn.Module):
    def __init__(self, num_cls: int, input_dim: int = 3) -> None:
        super().__init__()
        self.point_net_base = PointNetBase(input_dim)
        self.shared_layer_one = SharedMLP(
            1088, [512, 256, 128], use_batch_norm=True, use_relu=True
        )
        self.shared_layer_two = SharedMLP(
            128, [num_cls], use_batch_norm=True, use_relu=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, out_feature_trms, out_max_pool = self.point_net_base(x)
        out_max_pool_expend = out_max_pool.repeat(1, 1, out_feature_trms.shape[-1])
        concat = torch.concat((out_feature_trms, out_max_pool_expend), dim=1)
        x = self.shared_layer_one(concat)
        return self.shared_layer_two(x).transpose(2, 1).contiguous()


if __name__ == "__main__":
    # a = PointNetBase()
    # x = torch.randn((2, 3, 64))
    # out = a(x)
    # print(out.shape)
    # b = TNet(64)
    # x = torch.randn((2, 64, 64))
    # out = b(x)
    # print(out.shape)

    a = PointNetSeg(10)
    sample = torch.randn((2, 3, 64))
    out = a(sample)
    print(out.shape)
