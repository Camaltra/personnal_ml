from dataclasses import dataclass


@dataclass
class ViTConfig:
    num_layer_transformer: int
    embed_dim: int
    mlp_hidden_size: int
    dropout_linear: float
    dropout_embedding: float
    num_of_head: int
    patch_size: int
    batch_size: int
    image_size: tuple[int, int]
    num_class: int
    adam_beta_1: float
    adam_beta_2: float
    learning_rate: float
    num_epoch: int

    def print_config(self) -> None:
        print(
            f"DATA BASED CONFIG:\n\tBATCH SIZE: {self.batch_size}\n\tPATCH SIZE: {self.patch_size}\n\tIMAGE SIZE: {self.image_size}\n\tNUMBER OF CLASS: {self.num_class}"
        )
        print(
            f"MODEL BASED CONFIG\n\tNUMBER OF ENCODER TRANSFORMER: {self.num_layer_transformer}\n\tEMBEDDING SIZE: {self.embed_dim}\n\tDROPOUT EMBEDDING: {self.dropout_embedding}\n\tDROPOUT LINEAR: {self.dropout_linear}\n\tMLP HIDDEN SIZE: {self.mlp_hidden_size}\n\tNUMBER OF HEADS: {self.num_of_head}"
        )
        print(
            f"OPTIMIZER ADAM\n\tBETA 1: {self.adam_beta_1}\n\tBETA 2: {self.adam_beta_2}\n\tLEARNING_RATE: {self.learning_rate}"
        )
        print(f"TRAINING\n\tEPOCH: {self.num_epoch}")


BASEVITCONFIG = ViTConfig(
    num_layer_transformer=12,
    embed_dim=768,
    mlp_hidden_size=3072,
    dropout_linear=0.1,
    dropout_embedding=0.1,
    num_of_head=12,
    patch_size=16,
    batch_size=32,
    image_size=(512, 512),
    num_class=1000,
    adam_beta_1=0.9,
    adam_beta_2=0.999,
    learning_rate=2e-4,
    num_epoch=100,
)

XSMALLVITCONFIG = ViTConfig(
    num_layer_transformer=2,
    embed_dim=768,
    mlp_hidden_size=768,
    dropout_linear=0.1,
    dropout_embedding=0.1,
    num_of_head=12,
    patch_size=16,
    batch_size=128,
    image_size=(256, 256),
    num_class=1,
    adam_beta_1=0.9,
    adam_beta_2=0.999,
    learning_rate=1e-4,  # Lower batch size than the Base version, so get a smaller LR
    num_epoch=10,
)