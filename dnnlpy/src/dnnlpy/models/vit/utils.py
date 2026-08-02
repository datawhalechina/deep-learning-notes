from torch import Tensor

__all__ = ['patchify']


def patchify(x: Tensor, patch_size: int) -> Tensor:
    """Split images into flattened patches.

    Args:
        x: Input images of shape `(batch_size, channels, height, width)`.
        patch_size: Height and width of each square patch.

    Returns:
        Tensor: Flattened patches of shape `(batch_size, num_patches, channels *
        patch_size * patch_size)`.
    """
    batch_size, _, height, width = x.size()
    if height % patch_size != 0 or width % patch_size != 0:
        raise AssertionError(
            'Image height and width must be divisible by `patch_size`.'
        )

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    x = x.unfold(2, patch_size, patch_size)
    x = x.unfold(3, patch_size, patch_size)
    x = x.permute(0, 2, 3, 1, 4, 5)
    # We can't use `view` here because the tensor is not contiguous after `permute`.
    x = x.reshape(batch_size, num_patches_h * num_patches_w, -1)
    return x
