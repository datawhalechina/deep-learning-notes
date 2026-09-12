import torch

from . import (
    cs336 as cs336,
    models as models,
    nn as nn,
    optim as optim,
    tokenizers as tokenizers,
)
from .configtools import (
    get_data_root as get_data_root,
    get_default_device as get_default_device,
    get_num_workers as get_num_workers,
    has_gil as has_gil,
    set_seed as set_seed,
)
from .pylabtools import set_matplotlib_format as set_matplotlib_format
from .trainingtools import Trainer as Trainer
from .utils import (
    bytes_to_gib as bytes_to_gib,
    bytes_to_mib as bytes_to_mib,
    count_params as count_params,
)


def is_avx2_vnni_2() -> bool:
    """Check if the current CPU supports AVX2, VNNI, and related features.

    See https://github.com/pytorch/pytorch/issues/148861 for more details.
    Code ported from https://github.com/intel/torch-xpu-ops/pull/4341.
    """
    cap = torch.cpu.get_capabilities()
    return (
        cap.get('avx2', False)
        and cap.get('avx_vnni', False)
        and cap.get('avx_vnni_int8', False)
        and cap.get('avx_ne_convert', False)
        and not cap.get('avx512_f', False)
    )


IS_AVX2_VNNI_2 = is_avx2_vnni_2()
