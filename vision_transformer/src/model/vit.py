import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ViTConfig


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_hidden_size: int, dropout_p: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=mlp_hidden_size, out_features=embed_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MSA(nn.Module):
    def __init__(self, embed_dim: int = 768, num_of_head: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_of_head = num_of_head
        self.depth = self.embed_dim // self.num_of_head
        self.Wq = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.Wk = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.Wv = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = x.shape
        return x.reshape(batch_size, -1, self.num_of_head, self.depth).permute(
            0, 2, 1, 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        qk = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim**0.5)

        weight = F.softmax(qk, dim=1)

        attention = torch.matmul(weight, v)

        attention = attention.permute(0, 2, 1, 3).reshape(
            batch_size, -1, self.embed_dim
        )
        attention = self.linear(attention)
        return attention, weight


class Embeding(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        embed_dim: int = 768,
        patch_size: int = 16,
        batch_size: int = 32,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        num_of_patches = int((image_size[0] * image_size[1]) / patch_size**2)
        self.patches_layer = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        self.class_tokens = nn.Parameter(
            torch.ones(batch_size, 1, embed_dim), requires_grad=True
        )

        self.position_tokens = nn.Parameter(
            torch.ones(batch_size, num_of_patches + 1, embed_dim), requires_grad=True
        )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patches_layer(x)
        patches = nn.Flatten(start_dim=2, end_dim=3)(patches).permute(0, 2, 1)
        patches = torch.concat((self.class_tokens, patches), dim=1)
        patches += self.position_tokens
        return self.dropout(patches)


class TransformerEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, num_of_head: int, mlp_hidden_size: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.layer_norm_before_msa = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = MSA(embed_dim=embed_dim, num_of_head=num_of_head)

        self.layer_norm_before_mlp = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim, mlp_hidden_size=mlp_hidden_size, dropout_p=dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        msa_output, _ = self.msa(self.layer_norm_before_msa(x))
        mlp_output = self.mlp(self.layer_norm_before_mlp(msa_output + x))
        return mlp_output + msa_output


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.embeding_layer = Embeding(
            image_size=cfg.image_size,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size,
            batch_size=cfg.batch_size,
            dropout_p=cfg.dropout_embedding,
        )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embed_dim=cfg.embed_dim,
                    num_of_head=cfg.num_of_head,
                    mlp_hidden_size=cfg.mlp_hidden_size,
                    dropout_p=cfg.dropout_linear,
                )
                for _ in range(cfg.num_layer_transformer)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=cfg.embed_dim),
            nn.Linear(in_features=cfg.embed_dim, out_features=cfg.num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded_imgs = self.embeding_layer(x)
        return self.classifier(self.transformer_encoder(embedded_imgs)[:, 0])
