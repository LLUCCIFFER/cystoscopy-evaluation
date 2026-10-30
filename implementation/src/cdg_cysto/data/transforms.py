"""Image transforms for baseline and final CDG-Cysto training."""

from __future__ import annotations

from typing import Any


def build_transforms(
    *,
    train: bool,
    input_size: int = 224,
    policy: str = "baseline",
) -> object:
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        return build_torchvision_transforms(train=train, input_size=input_size, policy=policy)

    transforms = [A.Resize(input_size, input_size)]
    if train:
        transforms.append(A.HorizontalFlip(p=0.5))
        if policy == "final":
            transforms.extend(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.6),
                    A.Rotate(limit=15, border_mode=0, p=0.5),
                    A.AugMix(severity=2, width=3, depth=-1, p=0.5),
                ]
            )

    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


class TorchvisionTransform:
    """Albumentations-compatible wrapper around torchvision transforms."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, *, image: Any) -> dict[str, Any]:
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return {"image": self.transform(image)}


def build_torchvision_transforms(
    *,
    train: bool,
    input_size: int = 224,
    policy: str = "baseline",
) -> TorchvisionTransform:
    """Build torchvision fallback transforms for Windows DLL edge cases."""

    from torchvision import transforms

    ops: list[Any] = [transforms.Resize((input_size, input_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if policy == "final":
            ops.extend(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.15,
                        hue=0.05,
                    ),
                    transforms.RandomRotation(degrees=15),
                ]
            )
            augmix = getattr(transforms, "AugMix", None)
            if augmix is not None:
                ops.append(augmix(severity=2, mixture_width=3))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return TorchvisionTransform(transforms.Compose(ops))
