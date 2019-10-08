import torch
import torch.nn as nn
from targeted_dropout import TargetedDropout


def main():
    x = torch.rand(5, 2)

    targeted_weight_dropout = TargetedDropout(
        targeted="weight",
        target_layer=nn.Linear(2, 10),
        dropout_rate=0.5,
        targeted_portion=0.5,
    )

    output = targeted_weight_dropout(x)


if __name__ == "__main__":
    main()
