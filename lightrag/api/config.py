"""
Configs for the LightRAG API.
"""

import os
import re
import argparse
import logging
from lightrag.config import NO_PREFIX_SENTINEL, settings
from lightrag.utils import logger
from lightrag.llm.binding_options import (
    GeminiEmbeddingOptions,
    GeminiLLMOptions,
    OllamaEmbeddingOptions,
    OllamaLLMOptions,
    OpenAILLMOptions,
)
from lightrag.base import OllamaServerInfos
import sys

from lightrag.constants import (
    DEFAULT_MAX_ASYNC,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_RERANK_BINDING,
    DEFAULT_RERANK_MAX_ASYNC,
    DEFAULT_RERANK_TIMEOUT,
)


ollama_server_infos = OllamaServerInfos()
DEFAULT_TOKEN_SECRET = "lightrag-jwt-default-secret-key!"
PROVIDER_ASYMMETRIC_EMBEDDING_BINDINGS = {"gemini", "jina", "voyageai"}
PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS = {"azure_openai", "ollama", "openai"}


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "NetworkXStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


def get_default_host(binding_type: str) -> str:
    return settings.default_host_for_binding(binding_type)


def resolve_asymmetric_embedding_opt_in(
    *,
    binding: str,
    embedding_asymmetric: bool,
    embedding_asymmetric_configured: bool,
    query_prefix: str | None,
    document_prefix: str | None,
    query_prefix_configured: bool = False,
    document_prefix_configured: bool = False,
) -> bool:
    """Resolve whether query/document-aware embedding behavior should be enabled."""
    has_non_empty_prefix = bool(query_prefix or document_prefix)
    has_prefix_config = query_prefix_configured or document_prefix_configured

    if not embedding_asymmetric:
        if has_prefix_config:
            state = "false" if embedding_asymmetric_configured else "unset"
            logger.warning(
                f"EMBEDDING_ASYMMETRIC is {state}; "
                "EMBEDDING_QUERY_PREFIX and EMBEDDING_DOCUMENT_PREFIX will be ignored."
            )
        return False

    if binding in PROVIDER_ASYMMETRIC_EMBEDDING_BINDINGS:
        if has_prefix_config:
            logger.warning(
                f"{binding} embeddings use provider task parameters for asymmetric "
                "mode; EMBEDDING_QUERY_PREFIX and EMBEDDING_DOCUMENT_PREFIX will be ignored."
            )
        return True

    if binding in PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS:
        if not query_prefix_configured or not document_prefix_configured:
            raise ValueError(
                f"EMBEDDING_ASYMMETRIC=true for {binding} embeddings requires both "
                "EMBEDDING_QUERY_PREFIX and EMBEDDING_DOCUMENT_PREFIX. Use "
                f"{NO_PREFIX_SENTINEL} for a side that should intentionally have no prefix."
            )

        if not has_non_empty_prefix:
            raise ValueError(
                "At least one of EMBEDDING_QUERY_PREFIX or EMBEDDING_DOCUMENT_PREFIX "
                f"must be non-empty. Use {NO_PREFIX_SENTINEL} only for the side that "
                "should intentionally have no prefix."
            )
        return True

    raise ValueError(
        f"EMBEDDING_ASYMMETRIC=true is not supported for {binding} embeddings."
    )

def validate_auth_configuration(args: argparse.Namespace) -> None:
    """Reject insecure JWT auth settings before the API starts."""
    auth_accounts = (getattr(args, "auth_accounts", "") or "").strip()
    token_secret = (getattr(args, "token_secret", "") or "").strip()

    if auth_accounts and (not token_secret or token_secret == DEFAULT_TOKEN_SECRET):
        raise ValueError(
            "TOKEN_SECRET must be explicitly set to a non-default value when AUTH_ACCOUNTS is configured."
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Args:
        is_uvicorn_mode: Whether running under uvicorn mode

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="LightRAG API Server")

    # Server configuration
    parser.add_argument(
        "--host",
        default=settings.host,
        help="Server host (default: from env or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Server port (default: from env or 9621)",
    )

    # Directory configuration
    parser.add_argument(
        "--working-dir",
        default=settings.working_dir,
        help="Working directory for RAG storage (default: from env or ./rag_storage)",
    )
    parser.add_argument(
        "--input-dir",
        default=settings.input_dir,
        help="Directory containing input documents (default: from env or ./inputs)",
    )

    parser.add_argument(
        "--timeout",
        default=settings.timeout,
        type=int,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout",
    )

    # RAG configuration
    parser.add_argument(
        "--max-async",
        type=int,
        default=settings.max_async,
        help=f"Maximum async operations (default: from env or {DEFAULT_MAX_ASYNC})",
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=settings.summary_max_tokens,
        help=f"Maximum token size for entity/relation summary(default: from env or {DEFAULT_SUMMARY_MAX_TOKENS})",
    )
    parser.add_argument(
        "--summary-context-size",
        type=int,
        default=settings.summary_context_size,
        help=f"LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_CONTEXT_SIZE})",
    )
    parser.add_argument(
        "--summary-length-recommended",
        type=int,
        default=settings.summary_length_recommended,
        help=f"LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_LENGTH_RECOMMENDED})",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from env or INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=settings.verbose,
        help="Enable verbose debug output(only valid for DEBUG log-level)",
    )

    parser.add_argument(
        "--key",
        type=str,
        default=settings.lightrag_api_key,
        help="API key for authentication. This protects lightrag server against unauthorized access",
    )

    # Optional https parameters
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=settings.ssl,
        help="Enable HTTPS (default: from env or False)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=settings.ssl_certfile,
        help="Path to SSL certificate file (required if --ssl is enabled)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=settings.ssl_keyfile,
        help="Path to SSL private key file (required if --ssl is enabled)",
    )

    # Ollama model configuration
    parser.add_argument(
        "--simulated-model-name",
        type=str,
        default=settings.ollama_emulating_model_name,
        help="Name for the simulated Ollama model (default: from env or lightrag)",
    )

    parser.add_argument(
        "--simulated-model-tag",
        type=str,
        default=settings.ollama_emulating_model_tag,
        help="Tag for the simulated Ollama model (default: from env or latest)",
    )

    # Namespace
    parser.add_argument(
        "--workspace",
        type=str,
        default=settings.workspace,
        help="Default workspace for all storage",
    )

    # Server workers configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help="Number of worker processes (default: from env or 1)",
    )

    # LLM and embedding bindings
    parser.add_argument(
        "--llm-binding",
        type=str,
        default=settings.llm_binding,
        choices=[
            "lollms",
            "ollama",
            "openai",
            "openai-ollama",
            "azure_openai",
            "aws_bedrock",
            "gemini",
        ],
        help="LLM binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--embedding-binding",
        type=str,
        default=settings.embedding_binding,
        choices=[
            "lollms",
            "ollama",
            "openai",
            "azure_openai",
            "aws_bedrock",
            "jina",
            "gemini",
            "voyageai",
        ],
        help="Embedding binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--rerank-binding",
        type=str,
        default=settings.rerank_binding,
        choices=["null", "cohere", "jina", "aliyun"],
        help=f"Rerank binding type (default: from env or {DEFAULT_RERANK_BINDING})",
    )

    # Document loading engine configuration
    parser.add_argument(
        "--docling",
        action="store_true",
        default=False,
        help="Enable DOCLING document loading engine (default: from env or DEFAULT)",
    )

    # Conditionally add binding-specific options (Ollama, OpenAI, Azure OpenAI, Gemini)
    # This registers command line arguments (e.g., --openai-llm-temperature)
    # and reads corresponding environment variables (e.g., OPENAI_LLM_TEMPERATURE)

    # Determine LLM binding value consistently from command line or environment
    llm_binding_value = None
    if "--llm-binding" in sys.argv:
        try:
            idx = sys.argv.index("--llm-binding")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                llm_binding_value = sys.argv[idx + 1]
        except IndexError:
            pass

    # Fall back to environment variable using same function as argparse default
    if llm_binding_value is None:
        llm_binding_value = settings.llm_binding

    # Add LLM binding options based on determined value
    if llm_binding_value == "ollama":
        OllamaLLMOptions.add_args(parser)
    elif llm_binding_value in ["openai", "azure_openai"]:
        OpenAILLMOptions.add_args(parser)
    elif llm_binding_value == "gemini":
        GeminiLLMOptions.add_args(parser)

    # Determine embedding binding value consistently from command line or environment
    embedding_binding_value = None
    if "--embedding-binding" in sys.argv:
        try:
            idx = sys.argv.index("--embedding-binding")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                embedding_binding_value = sys.argv[idx + 1]
        except IndexError:
            pass

    # Fall back to environment variable using same function as argparse default
    if embedding_binding_value is None:
        embedding_binding_value = settings.embedding_binding

    # Add embedding binding options based on determined value
    if embedding_binding_value == "ollama":
        OllamaEmbeddingOptions.add_args(parser)
    elif embedding_binding_value == "gemini":
        GeminiEmbeddingOptions.add_args(parser)

    args = parser.parse_args()

    # convert relative path to absolute path
    args.working_dir = os.path.abspath(args.working_dir)
    args.input_dir = os.path.abspath(args.input_dir)

    # Inject storage configuration from environment variables
    args.kv_storage = settings.lightrag_kv_storage
    args.doc_status_storage = settings.lightrag_doc_status_storage
    args.graph_storage = settings.lightrag_graph_storage
    args.vector_storage = settings.lightrag_vector_storage

    # Get MAX_PARALLEL_INSERT from environment
    args.max_parallel_insert = settings.max_parallel_insert

    # Get MAX_GRAPH_NODES from environment
    args.max_graph_nodes = settings.max_graph_nodes

    # Handle openai-ollama special case
    if args.llm_binding == "openai-ollama":
        args.llm_binding = "openai"
        args.embedding_binding = "ollama"

    args.llm_binding_host = settings.effective_llm_binding_host(args.llm_binding)
    args.embedding_binding_host = settings.effective_embedding_binding_host(
        args.embedding_binding
    )
    args.llm_binding_api_key = settings.llm_binding_api_key
    args.embedding_binding_api_key = settings.embedding_binding_api_key or ""

    # Inject model configuration
    args.llm_model = settings.llm_model
    # EMBEDDING_MODEL defaults to None - each binding will use its own default model
    # e.g., OpenAI uses "text-embedding-3-small", Jina uses "jina-embeddings-v4"
    args.embedding_model = settings.embedding_model
    # EMBEDDING_DIM defaults to None - each binding will use its own default dimension
    # Value is inherited from provider defaults via wrap_embedding_func_with_attrs decorator
    args.embedding_dim = settings.embedding_dim
    args.embedding_send_dim = settings.embedding_send_dim

    # Inject chunk configuration
    args.chunk_size = settings.chunk_size
    args.chunk_overlap_size = settings.chunk_overlap_size

    # Inject LLM cache configuration
    args.enable_llm_cache_for_extract = settings.enable_llm_cache_for_extract
    args.enable_llm_cache = settings.enable_llm_cache

    # Set document_loading_engine from --docling flag
    if args.docling:
        args.document_loading_engine = "DOCLING"
    else:
        args.document_loading_engine = settings.document_loading_engine

    # PDF decryption password
    args.pdf_decrypt_password = settings.pdf_decrypt_password

    # Add environment variables that were previously read directly
    args.cors_origins = settings.cors_origins
    args.summary_language = settings.summary_language
    args.entity_types = settings.entity_types
    args.relation_labels = settings.relation_labels
    args.whitelist_paths = settings.whitelist_paths

    # For JWT Auth
    args.auth_accounts = settings.auth_accounts
    args.token_secret = settings.token_secret
    args.token_expire_hours = settings.token_expire_hours
    args.guest_token_expire_hours = settings.guest_token_expire_hours
    args.jwt_algorithm = settings.jwt_algorithm

    # Token auto-renewal configuration (sliding window expiration)
    args.token_auto_renew = settings.token_auto_renew
    args.token_renew_threshold = settings.token_renew_threshold

    # Rerank model configuration
    args.rerank_model = settings.rerank_model
    args.rerank_binding_host = settings.rerank_binding_host
    args.rerank_binding_api_key = settings.rerank_binding_api_key
    # Note: rerank_binding is already set by argparse, no need to override from env

    # Min rerank score configuration
    args.min_rerank_score = settings.min_rerank_score

    # Query configuration
    args.history_turns = settings.history_turns
    args.top_k = settings.top_k
    args.chunk_top_k = settings.chunk_top_k
    args.max_entity_tokens = settings.max_entity_tokens
    args.max_relation_tokens = settings.max_relation_tokens
    args.max_total_tokens = settings.max_total_tokens
    args.cosine_threshold = settings.cosine_threshold
    args.related_chunk_number = settings.related_chunk_number

    # Add missing environment variables for health endpoint
    args.force_llm_summary_on_merge = settings.force_llm_summary_on_merge
    args.embedding_func_max_async = settings.embedding_func_max_async
    args.embedding_batch_num = settings.embedding_batch_num

    # Embedding token limit configuration
    args.embedding_token_limit = settings.embedding_token_limit

    # File upload size limit (in bytes, None for unlimited)
    # Default: 100MB (104857600 bytes)
    args.max_upload_size = settings.max_upload_size

    # VLM multimodal processing toggle
    args.vlm_process_enable = (
        os.getenv("VLM_PROCESS_ENABLE", "false").lower() in ("true", "1", "yes", "t", "on")
    )

    # LLM/embedding timeout from env (used in health endpoint)
    args.llm_timeout = settings.llm_timeout
    args.embedding_timeout = settings.embedding_timeout

    # Rerank configuration from env
    args.rerank_max_async = DEFAULT_RERANK_MAX_ASYNC
    args.rerank_timeout = DEFAULT_RERANK_TIMEOUT

    # Embedding prefix configuration for context-aware embeddings. Empty prefixes
    # must be explicit via NO_PREFIX so missing config is distinguishable.
    args.embedding_document_prefix = settings.embedding_document_prefix
    args.embedding_document_prefix_configured = (
        settings.embedding_document_prefix_configured
    )
    args.embedding_query_prefix = settings.embedding_query_prefix
    args.embedding_query_prefix_configured = settings.embedding_query_prefix_configured
    args.embedding_prefix_no_prefix_sentinel = NO_PREFIX_SENTINEL
    args.embedding_prefixes_configured = settings.embedding_prefixes_configured
    # Asymmetric embedding behavior toggle
    args.embedding_asymmetric_configured = settings.embedding_asymmetric_configured
    args.embedding_asymmetric = settings.embedding_asymmetric

    ollama_server_infos.LIGHTRAG_NAME = args.simulated_model_name
    ollama_server_infos.LIGHTRAG_TAG = args.simulated_model_tag

    # Sanitize workspace: only alphanumeric characters and underscores are allowed
    if args.workspace:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", args.workspace)
        if sanitized != args.workspace:
            logging.warning(
                f"Workspace name '{args.workspace}' contains invalid characters. "
                f"It has been sanitized to '{sanitized}'. "
                "Only alphanumeric characters and underscores are allowed."
            )
            args.workspace = sanitized

    validate_auth_configuration(args)
    return args


def update_uvicorn_mode_config():
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    if global_args.workers > 1:
        original_workers = global_args.workers
        global_args.workers = 1
        # Log warning directly here
        logging.debug(
            f">> Forcing workers=1 in uvicorn mode(Ignoring workers={original_workers})"
        )


# Global configuration with lazy initialization
_global_args = None
_initialized = False


def initialize_config(args=None, force=False):
    """Initialize global configuration

    This function allows explicit initialization of the configuration,
    which is useful for programmatic usage, testing, or embedding LightRAG
    in other applications.

    Args:
        args: Pre-parsed argparse.Namespace or None to parse from sys.argv
        force: Force re-initialization even if already initialized

    Returns:
        argparse.Namespace: The configured arguments

    Example:
        # Use parsed command line arguments (default)
        initialize_config()

        # Use custom configuration programmatically
        custom_args = argparse.Namespace(
            host='localhost',
            port=8080,
            working_dir='./custom_rag',
            # ... other config
        )
        initialize_config(custom_args)
    """
    global _global_args, _initialized

    if _initialized and not force:
        return _global_args

    resolved_args = args if args is not None else parse_args()
    validate_auth_configuration(resolved_args)
    _global_args = resolved_args
    _initialized = True
    return _global_args


def get_config():
    """Get global configuration, auto-initializing if needed

    Returns:
        argparse.Namespace: The configured arguments
    """
    if not _initialized:
        initialize_config()
    return _global_args


class _GlobalArgsProxy:
    """Proxy object that auto-initializes configuration on first access

    This maintains backward compatibility with existing code while
    allowing programmatic control over initialization timing.

    The proxy fully delegates to the underlying argparse.Namespace,
    including support for vars() calls which is used by binding_options
    to extract provider-specific configuration options.
    """

    def __getattribute__(self, name):
        """Override attribute access to support vars() and regular attribute access.

        This method intercepts __dict__ access (used by vars()) and delegates
        to the underlying _global_args namespace, ensuring binding options
        can be properly extracted.
        """
        global _initialized, _global_args

        # Handle __dict__ access for vars() support
        if name == "__dict__":
            if not _initialized:
                initialize_config()
            return vars(_global_args)

        # Handle class-level attributes that should come from the proxy itself
        if name in ("__class__", "__repr__", "__getattribute__", "__setattr__"):
            return object.__getattribute__(self, name)

        # Delegate all other attribute access to the underlying namespace
        if not _initialized:
            initialize_config()
        return getattr(_global_args, name)

    def __setattr__(self, name, value):
        global _initialized, _global_args
        if not _initialized:
            initialize_config()
        setattr(_global_args, name, value)

    def __repr__(self):
        global _initialized, _global_args
        if not _initialized:
            return "<GlobalArgsProxy: Not initialized>"
        return repr(_global_args)


# Create proxy instance for backward compatibility
# Existing code like `from config import global_args` continues to work
# The proxy will auto-initialize on first attribute access
global_args = _GlobalArgsProxy()
