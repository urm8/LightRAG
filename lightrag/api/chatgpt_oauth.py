"""OAuth 2.1 provider for the ChatGPT-facing LightRAG MCP endpoint."""

from __future__ import annotations

import asyncio
import html
import secrets
import time
from collections.abc import Callable
from urllib.parse import quote

import jwt
from fastmcp.server.auth import AccessToken, OAuthProvider
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Route

READ_SCOPE = "memory:read"
WRITE_SCOPE = "memory:write"
SUPPORTED_SCOPES = [READ_SCOPE, WRITE_SCOPE]


class UserAuthorizationCode(AuthorizationCode):
    username: str
    jti: str
    resource: str | None = None


class UserRefreshToken(RefreshToken):
    username: str
    resource: str
    jti: str


class LightRAGChatGPTOAuthProvider(OAuthProvider):
    """Small OAuth authorization server backed by existing LightRAG accounts."""

    def __init__(
        self,
        base_url: str,
        secret: str,
        password_verifier: Callable[[str, str], bool],
        algorithm: str = "HS256",
    ):
        super().__init__(
            base_url=base_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=SUPPORTED_SCOPES,
                default_scopes=SUPPORTED_SCOPES,
            ),
            required_scopes=[READ_SCOPE],
        )
        self.secret = secret
        self.algorithm = algorithm
        self.password_verifier = password_verifier
        self.resource = f"{base_url.rstrip('/')}/mcp"
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.used_codes: set[str] = set()
        self.revoked_refresh_tokens: set[str] = set()

    def _encode(self, payload: dict, expires_in: int) -> str:
        now = int(time.time())
        return jwt.encode(
            {**payload, "iat": now, "exp": now + expires_in},
            self.secret,
            algorithm=self.algorithm,
        )

    def _decode(self, token: str, token_type: str) -> dict | None:
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                options={"verify_aud": False},
            )
        except jwt.PyJWTError:
            return None
        return payload if payload.get("type") == token_type else None

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if client_info.client_id is None:
            raise RegistrationError("invalid_client_metadata", "client_id is required")
        self.clients[client_info.client_id] = client_info

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if params.resource != self.resource:
            raise AuthorizeError("invalid_request", "Invalid OAuth resource")
        scopes = params.scopes or [READ_SCOPE]
        if not set(scopes).issubset(SUPPORTED_SCOPES):
            raise AuthorizeError("invalid_scope", "Unsupported memory scope")
        request_token = self._encode(
            {
                "type": "authorization_request",
                "client_id": client.client_id,
                "redirect_uri": str(params.redirect_uri),
                "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
                "state": params.state,
                "scopes": scopes,
                "code_challenge": params.code_challenge,
                "resource": params.resource,
            },
            300,
        )
        return f"{str(self.base_url).rstrip('/')}/login?request={quote(request_token)}"

    async def _login_page(self, request: Request) -> HTMLResponse:
        request_token = request.query_params.get("request", "")
        if self._decode(request_token, "authorization_request") is None:
            return HTMLResponse("Invalid or expired authorization request", status_code=400)
        escaped = html.escape(request_token, quote=True)
        return HTMLResponse(
            f"""<!doctype html><html><head><meta name="viewport" content="width=device-width">
            <title>Connect LightRAG</title><style>body{{font:16px system-ui;max-width:28rem;margin:4rem auto;padding:1rem}}
            label{{display:block;margin-top:1rem}}input{{box-sizing:border-box;width:100%;padding:.7rem}}
            button{{margin-top:1.2rem;padding:.75rem 1rem}}</style></head><body>
            <h1>Connect LightRAG memory</h1><p>Sign in to let ChatGPT search and save your LightRAG memory.</p>
            <form method="post"><input type="hidden" name="request" value="{escaped}">
            <label>Username<input name="username" autocomplete="username" required></label>
            <label>Password<input type="password" name="password" autocomplete="current-password" required></label>
            <button type="submit">Authorize ChatGPT</button></form></body></html>"""
        )

    async def _login(self, request: Request):
        form = await request.form()
        request_token = str(form.get("request", ""))
        payload = self._decode(request_token, "authorization_request")
        if payload is None:
            return HTMLResponse("Invalid or expired authorization request", status_code=400)
        username = str(form.get("username", ""))
        password = str(form.get("password", ""))
        if not await asyncio.to_thread(self.password_verifier, username, password):
            return HTMLResponse("Invalid username or password", status_code=401)

        code = self._encode(
            {
                "type": "authorization_code",
                "client_id": payload["client_id"],
                "redirect_uri": payload["redirect_uri"],
                "redirect_uri_provided_explicitly": payload[
                    "redirect_uri_provided_explicitly"
                ],
                "scopes": payload["scopes"],
                "code_challenge": payload["code_challenge"],
                "resource": payload["resource"],
                "username": username,
                "jti": secrets.token_urlsafe(24),
            },
            180,
        )
        return RedirectResponse(
            construct_redirect_uri(
                payload["redirect_uri"], code=code, state=payload.get("state")
            ),
            status_code=302,
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> UserAuthorizationCode | None:
        payload = self._decode(authorization_code, "authorization_code")
        if (
            payload is None
            or payload.get("client_id") != client.client_id
            or payload.get("jti") in self.used_codes
        ):
            return None
        return UserAuthorizationCode(
            code=authorization_code,
            client_id=payload["client_id"],
            redirect_uri=payload["redirect_uri"],
            redirect_uri_provided_explicitly=payload[
                "redirect_uri_provided_explicitly"
            ],
            scopes=payload["scopes"],
            expires_at=payload["exp"],
            code_challenge=payload["code_challenge"],
            resource=payload["resource"],
            username=payload["username"],
            jti=payload["jti"],
        )

    def _issue_tokens(
        self, client_id: str, username: str, scopes: list[str], resource: str
    ) -> OAuthToken:
        access_token = self._encode(
            {
                "type": "access_token",
                "sub": username,
                "client_id": client_id,
                "scope": " ".join(scopes),
                "aud": resource,
                "iss": str(self.issuer_url),
            },
            3600,
        )
        refresh_token = self._encode(
            {
                "type": "refresh_token",
                "sub": username,
                "client_id": client_id,
                "scopes": scopes,
                "resource": resource,
                "jti": secrets.token_urlsafe(24),
            },
            30 * 24 * 3600,
        )
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token,
            scope=" ".join(scopes),
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: UserAuthorizationCode
    ) -> OAuthToken:
        if authorization_code.jti in self.used_codes:
            raise TokenError("invalid_grant", "Authorization code already used")
        self.used_codes.add(authorization_code.jti)
        return self._issue_tokens(
            client.client_id or "",
            authorization_code.username,
            authorization_code.scopes,
            authorization_code.resource or self.resource,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        payload = self._decode(token, "access_token")
        if payload is None or payload.get("aud") != self.resource:
            return None
        scopes = str(payload.get("scope", "")).split()
        return AccessToken(
            token=token,
            client_id=payload.get("client_id", ""),
            scopes=scopes,
            expires_at=payload["exp"],
            resource=self.resource,
            claims={"sub": payload.get("sub", "")},
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> UserRefreshToken | None:
        payload = self._decode(refresh_token, "refresh_token")
        if (
            payload is None
            or payload.get("client_id") != client.client_id
            or payload.get("jti") in self.revoked_refresh_tokens
        ):
            return None
        return UserRefreshToken(
            token=refresh_token,
            client_id=payload["client_id"],
            scopes=payload["scopes"],
            expires_at=payload["exp"],
            username=payload["sub"],
            resource=payload["resource"],
            jti=payload["jti"],
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: UserRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        if not set(scopes).issubset(refresh_token.scopes):
            raise TokenError("invalid_scope", "Requested scopes exceed grant")
        self.revoked_refresh_tokens.add(refresh_token.jti)
        return self._issue_tokens(
            client.client_id or "",
            refresh_token.username,
            scopes,
            refresh_token.resource,
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, UserRefreshToken):
            self.revoked_refresh_tokens.add(token.jti)

    def get_routes(self, mcp_path: str | None = None) -> list[Route]:
        return [
            *super().get_routes(mcp_path),
            Route("/login", self._login_page, methods=["GET"]),
            Route("/login", self._login, methods=["POST"]),
        ]
