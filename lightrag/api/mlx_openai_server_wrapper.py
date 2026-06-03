from __future__ import annotations

import asyncio
import inspect
import sys
from typing import Any


def _install_dill_batch_setitems_compatibility_patch(
    dill_module: Any, app_dill_module: Any
) -> bool:
    parent_method = getattr(getattr(dill_module, "Pickler", None), "_batch_setitems", None)
    child_pickler = getattr(app_dill_module, "Pickler", None)
    child_method = getattr(child_pickler, "_batch_setitems", None)
    if parent_method is None or child_method is None or child_pickler is None:
        return False

    parent_params = len(inspect.signature(parent_method).parameters)
    child_params = len(inspect.signature(child_method).parameters)
    if child_params >= parent_params:
        return False

    hasher = getattr(app_dill_module, "Hasher", None)

    def _patched_batch_setitems(self, items: Any, obj: Any = None) -> None:
        if getattr(self, "_legacy_no_dict_keys_sorting", False):
            if obj is None:
                parent_method(self, items)
            else:
                parent_method(self, items, obj)
            return
        try:
            sorted_items = sorted(items)
        except Exception:
            if hasher is None:
                sorted_items = list(items)
            else:
                sorted_items = sorted(items, key=lambda item: hasher.hash(item[0]))
        if obj is None:
            parent_method(self, sorted_items)
        else:
            parent_method(self, sorted_items, obj)

    child_pickler._batch_setitems = _patched_batch_setitems
    return True


def _apply_dill_batch_setitems_compatibility_patch() -> bool:
    import dill
    from app.utils import dill as app_dill

    return _install_dill_batch_setitems_compatibility_patch(dill, app_dill)


def _lightrag_handler_worker(*args: Any, **kwargs: Any) -> None:
    """Child-process entrypoint wrapper that reapplies local compatibility patches."""

    _apply_dill_batch_setitems_compatibility_patch()
    _apply_gemma4_same_thread_inference_patch()

    import app.core.handler_process as handler_process_module

    original = getattr(
        handler_process_module,
        "_lightrag_original_handler_worker",
        handler_process_module._handler_worker,
    )
    return original(*args, **kwargs)


def _apply_gemma4_same_thread_inference_patch() -> bool:
    """Force Gemma 4 requests onto the handler thread.

    Some Gemma 4 MLX conversions currently fail under ``mlx-openai-server``'s
    worker-threaded LM paths with ``There is no Stream(gpu, N) in current
    thread``. Native ``mlx_lm`` generation works when model load, cache
    creation, and token generation stay on the same thread, so we selectively
    bypass the background inference worker for Gemma 4 handlers.
    """

    from app.core.inference_worker import InferenceWorker
    import app.core.handler_process as handler_process_module
    from app.handler.mlx_lm import MLXLMHandler
    from loguru import logger

    if getattr(InferenceWorker, "_lightrag_gemma4_inline_patch", False):
        return False

    original_submit = InferenceWorker.submit
    original_submit_stream = InferenceWorker.submit_stream
    original_is_request_batchable = MLXLMHandler._is_request_batchable
    original_init = MLXLMHandler.__init__
    original_initialize = MLXLMHandler.initialize

    def _inline_enabled(worker: InferenceWorker) -> bool:
        return bool(getattr(worker, "_lightrag_inline", False))

    async def _patched_submit(self, func, *args, **kwargs):
        if _inline_enabled(self):
            return func(*args, **kwargs)
        return await original_submit(self, func, *args, **kwargs)

    def _patched_submit_stream(self, func, *args, **kwargs):
        if not _inline_enabled(self):
            return original_submit_stream(self, func, *args, **kwargs)

        async def _inline_stream():
            gen = None
            try:
                gen = func(*args, **kwargs)
                for item in gen:
                    yield item
                    await asyncio.sleep(0)
            finally:
                if gen is not None:
                    try:
                        gen.close()
                    except Exception:
                        pass

        return _inline_stream()

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        model_type = str(getattr(self.model, "model_type", "") or "").lower()
        model_path = str(getattr(self, "model_path", "") or "").lower()
        if model_type == "gemma4" or "gemma4" in model_path:
            setattr(self, "_lightrag_same_thread_inference", True)
            logger.warning(
                "LightRAG patch: forcing same-thread inference for Gemma 4 model path=%s",
                getattr(self, "model_path", ""),
            )

    async def _patched_initialize(self, queue_config=None):
        result = await original_initialize(self, queue_config)
        if getattr(self, "_lightrag_same_thread_inference", False):
            setattr(self.inference_worker, "_lightrag_inline", True)
        return result

    def _patched_is_request_batchable(self, request):
        if getattr(self, "_lightrag_same_thread_inference", False):
            return False
        return original_is_request_batchable(self, request)

    InferenceWorker.submit = _patched_submit
    InferenceWorker.submit_stream = _patched_submit_stream
    MLXLMHandler.__init__ = _patched_init
    MLXLMHandler.initialize = _patched_initialize
    MLXLMHandler._is_request_batchable = _patched_is_request_batchable
    if not hasattr(handler_process_module, "_lightrag_original_handler_worker"):
        handler_process_module._lightrag_original_handler_worker = (
            handler_process_module._handler_worker
        )
    handler_process_module._handler_worker = _lightrag_handler_worker
    InferenceWorker._lightrag_gemma4_inline_patch = True
    return True


def main(argv: list[str] | None = None) -> None:
    _apply_dill_batch_setitems_compatibility_patch()
    _apply_gemma4_same_thread_inference_patch()
    from app.cli import cli

    cli.main(args=list(sys.argv[1:] if argv is None else argv), prog_name="mlx-openai-server")


if __name__ == "__main__":
    main()
