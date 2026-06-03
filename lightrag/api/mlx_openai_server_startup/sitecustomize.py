from lightrag.api.mlx_openai_server_wrapper import (
    _apply_dill_batch_setitems_compatibility_patch,
    _apply_gemma4_same_thread_inference_patch,
)


_apply_dill_batch_setitems_compatibility_patch()
_apply_gemma4_same_thread_inference_patch()
