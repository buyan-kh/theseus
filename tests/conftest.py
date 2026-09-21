# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(0)
