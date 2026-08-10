import pytest
from fastapi import HTTPException

from scripts.managed_mlx_embeddings_server import _validate_embedding_dimensions


@pytest.mark.offline
def test_embedding_dimension_validation_uses_model_output_shape():
    _validate_embedding_dimensions(None, 768)
    _validate_embedding_dimensions(768, 768)

    with pytest.raises(HTTPException, match="returns 768 dimensions") as exc_info:
        _validate_embedding_dimensions(1024, 768)

    assert exc_info.value.status_code == 400
