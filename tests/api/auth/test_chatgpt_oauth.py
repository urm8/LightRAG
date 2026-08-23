import asyncio
import base64
import hashlib
from urllib.parse import parse_qs, urlparse

from fastmcp import FastMCP
from starlette.testclient import TestClient

from lightrag.api.chatgpt_oauth import LightRAGChatGPTOAuthProvider


def _pkce() -> tuple[str, str]:
    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    return verifier, challenge


def test_chatgpt_oauth_discovery_login_and_token_exchange():
    provider = LightRAGChatGPTOAuthProvider(
        base_url="https://rag.example/chatgpt",
        secret="test-secret-with-sufficient-entropy",
        password_verifier=lambda username, password: (username, password)
        == ("admin", "correct"),
    )
    mcp = FastMCP("test", auth=provider)

    @mcp.tool
    def ping() -> str:
        return "pong"

    client = TestClient(
        mcp.http_app(
            path="/mcp", json_response=True, stateless_http=True
        ),
        follow_redirects=False,
    )
    resource = "https://rag.example/chatgpt/mcp"

    metadata = client.get("/.well-known/oauth-authorization-server").json()
    assert metadata["code_challenge_methods_supported"] == ["S256"]
    assert metadata["registration_endpoint"].endswith("/register")

    registration = client.post(
        "/register",
        json={
            "redirect_uris": ["https://chatgpt.com/test-callback"],
            "token_endpoint_auth_method": "client_secret_post",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": "memory:read memory:write",
            "client_name": "ChatGPT test",
        },
    )
    assert registration.status_code == 201
    registered = registration.json()
    verifier, challenge = _pkce()

    authorize = client.get(
        "/authorize",
        params={
            "client_id": registered["client_id"],
            "redirect_uri": "https://chatgpt.com/test-callback",
            "response_type": "code",
            "scope": "memory:read memory:write",
            "state": "state-1",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": resource,
        },
    )
    assert authorize.status_code == 302
    login_url = authorize.headers["location"]
    login_parts = urlparse(login_url)
    request_token = parse_qs(login_parts.query)["request"][0]
    assert client.get(f"/login?{login_parts.query}").status_code == 200

    login = client.post(
        "/login",
        data={"request": request_token, "username": "admin", "password": "correct"},
    )
    assert login.status_code == 302
    callback = urlparse(login.headers["location"])
    callback_params = parse_qs(callback.query)
    assert callback_params["state"] == ["state-1"]

    token = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": callback_params["code"][0],
            "redirect_uri": "https://chatgpt.com/test-callback",
            "client_id": registered["client_id"],
            "client_secret": registered["client_secret"],
            "code_verifier": verifier,
            "resource": resource,
        },
    )
    assert token.status_code == 200
    access_token = token.json()["access_token"]

    verified = asyncio.run(provider.load_access_token(access_token))
    assert verified is not None
    assert verified.claims["sub"] == "admin"
    assert verified.scopes == ["memory:read", "memory:write"]

    unauthorized = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert unauthorized.status_code == 401
    assert "resource_metadata=" in unauthorized.headers["www-authenticate"]
