"""Run a forward pass with synthetic data only."""

import torch

from spar_mil import SPARMIL


def main() -> None:
    torch.manual_seed(0)
    model = SPARMIL(input_channels=16).eval()

    patches = torch.randn(2, 12, 16, 24, 24)
    coordinates = torch.rand(2, 12, 2)
    valid_mask = torch.ones(2, 12, dtype=torch.bool)
    valid_mask[1, -3:] = False

    with torch.inference_mode():
        output = model(patches, coordinates, valid_mask)

    assert torch.isfinite(output.logits).all()
    assert torch.allclose(output.attention.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.equal(output.attention[1, -3:], torch.zeros(3))

    print("logits:", tuple(output.logits.shape))
    print("attention:", tuple(output.attention.shape))
    print("instance_logits:", tuple(output.instance_logits.shape))
    print("attention sums:", output.attention.sum(dim=1))
    print("smoke test: OK")


if __name__ == "__main__":
    main()

