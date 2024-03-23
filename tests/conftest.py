import pytest

from fms.models.llama import LLaMA, LLaMAConfig


@pytest.fixture
def narrow_model(request):
    return LLaMA(
        LLaMAConfig(src_vocab_size=1, emb_dim=1, nheads=1, nlayers=request.param)
    )


@pytest.fixture
def narrow_model_factory(request):
    class NarrowModelFactory:
        def create(self):
            return LLaMA(
                LLaMAConfig(
                    src_vocab_size=1, emb_dim=1, nheads=1, nlayers=request.param
                )
            )

    return NarrowModelFactory()
