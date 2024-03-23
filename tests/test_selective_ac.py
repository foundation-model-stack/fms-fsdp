import pytest
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from fms_fsdp.policies import apply_fsdp_checkpointing


@pytest.mark.parametrize("narrow_model_factory", [15], indirect=True)
def test_selective_ac(narrow_model_factory):
    model = narrow_model_factory.create()
    apply_fsdp_checkpointing(model, 0)
    expected = [False] * 15
    assert [isinstance(block, CheckpointWrapper) for block in model.layers] == expected

    model = narrow_model_factory.create()
    apply_fsdp_checkpointing(model, 1 / 5)
    expected = [False, False, True, False, False] * 3
    assert [isinstance(block, CheckpointWrapper) for block in model.layers] == expected

    model = narrow_model_factory.create()
    apply_fsdp_checkpointing(model, 1 / 3)
    expected = [False, True, False] * 5
    assert [isinstance(block, CheckpointWrapper) for block in model.layers] == expected

    model = narrow_model_factory.create()
    apply_fsdp_checkpointing(model, 1 / 2)
    expected = [True, False] * 7 + [True]
    assert [isinstance(block, CheckpointWrapper) for block in model.layers] == expected

    model = narrow_model_factory.create()
    apply_fsdp_checkpointing(model, 3 / 5)
    expected = [True, False, True, False, True] * 3
    assert [isinstance(block, CheckpointWrapper) for block in model.layers] == expected

    model = narrow_model_factory.create()
    apply_fsdp_checkpointing(model, 2 / 3)
    expected = [True, False, True] * 5
    assert [isinstance(block, CheckpointWrapper) for block in model.layers] == expected

    model = narrow_model_factory.create()
    apply_fsdp_checkpointing(model, 1)
    expected = [True] * 15
    assert [isinstance(block, CheckpointWrapper) for block in model.layers] == expected
