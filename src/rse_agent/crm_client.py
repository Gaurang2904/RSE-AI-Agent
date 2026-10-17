"""HTTP client for the CRM REST API.

Integration patterns demonstrated:
- bearer-token authentication (token from RSE_CRM_TOKEN, never logged)
- connect/read timeouts on every request
- retries with exponential backoff + jitter on 5xx, timeouts and connection errors
- 429 handling that honours the server's Retry-After header (capped)
- automatic pagination (follows next_page, with a page cap)
- response validation with Pydantic: schema drift becomes a clear CRMValidationError
- typed exceptions with messages a user (or the agent) can act on
Only filters and row numbers are sent to the CRM, never client names or personal data.
"""
from __future__ import annotations

import os
import random
import time
from datetime import date
from typing import Generic, Iterator, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------- response models
class Product(BaseModel):
    product_id: str
    name: str
    category: str
    provider: str
    min_ticket_inr: int = Field(ge=0)
    risk_level: Literal["low", "medium", "high"]
    active: bool


class FollowUp(BaseModel):
    followup_id: str
    client_ref: int = Field(ge=1)
    product_category: str
    status: Literal["open", "done"]
    due_date: date
    owner: str
    note: str


class Page(BaseModel, Generic[T]):
    data: list[T]
    page: int
    page_size: int
    total: int
    next_page: int | None


# ---------------------------------------------------------------- errors
class CRMError(Exception):
    """Base class: str(err) is a message suitable to show the user."""


class CRMAuthError(CRMError):
    pass


class CRMNotFound(CRMError):
    pass


class CRMUnavailable(CRMError):
    pass


class CRMValidationError(CRMError):
    pass


# ---------------------------------------------------------------- client
class CRMClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, *, timeout: float = 5.0,
                 max_retries: int = 3, backoff: float = 0.5, max_retry_after: float = 10.0,
                 max_pages: int = 50, http: httpx.Client | None = None):
        self.base_url = (base_url or os.getenv("RSE_CRM_URL", "http://127.0.0.1:8765")).rstrip("/")
        self._token = token if token is not None else os.getenv("RSE_CRM_TOKEN", "demo-token")
        self.timeout = httpx.Timeout(timeout, connect=min(timeout, 3.0))
        self.max_retries, self.backoff = max_retries, backoff
        self.max_retry_after, self.max_pages = max_retry_after, max_pages
        self._http = http or httpx.Client(base_url=self.base_url, timeout=self.timeout)
        self.stats = {"requests": 0, "retries": 0, "pages": 0}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), self.max_retry_after))
                return
            except ValueError:
                pass
        time.sleep(self.backoff * (2 ** attempt) + random.uniform(0, self.backoff / 2))

    def _get(self, path: str, params: dict | None = None) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        last = "no response"
        for attempt in range(self.max_retries + 1):
            self.stats["requests"] += 1
            try:
                r = self._http.get(path, params=params, headers=self._headers(), timeout=self.timeout)
            except httpx.TimeoutException:
                last = f"timed out after {self.timeout.read}s"
            except httpx.TransportError as e:
                last = f"could not connect to the CRM at {self.base_url} ({type(e).__name__})"
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError as e:
                        raise CRMValidationError(f"CRM returned non-JSON data for {path}") from e
                if r.status_code in (401, 403):
                    raise CRMAuthError("CRM rejected the credentials (check RSE_CRM_TOKEN)")
                if r.status_code == 404:
                    raise CRMNotFound(f"CRM has no resource at {path}")
                if r.status_code == 429:
                    last = "rate limited by the CRM"
                    if attempt < self.max_retries:
                        self.stats["retries"] += 1
                        self._sleep(attempt, r.headers.get("Retry-After"))
                        continue
                    break
                if r.status_code >= 500:
                    last = f"CRM server error {r.status_code}"
                else:
                    raise CRMError(f"CRM request failed with HTTP {r.status_code}: {r.text[:200]}")
            if attempt < self.max_retries:
                self.stats["retries"] += 1
                self._sleep(attempt)
        raise CRMUnavailable(f"CRM unavailable after {self.max_retries + 1} attempts: {last}")

    def _pages(self, path: str, model: type[T], params: dict | None = None, page_size: int = 50) -> Iterator[T]:
        page = 1
        for _ in range(self.max_pages):
            raw = self._get(path, {**(params or {}), "page": page, "page_size": page_size})
            try:
                parsed = Page[model].model_validate(raw)
            except ValidationError as e:
                first = e.errors()[0]
                where = ".".join(str(p) for p in first["loc"])
                raise CRMValidationError(f"CRM returned unexpected data at '{where}': {first['msg']}") from e
            self.stats["pages"] += 1
            yield from parsed.data
            if parsed.next_page is None:
                return
            page = parsed.next_page
        raise CRMError(f"stopped after {self.max_pages} pages (pagination did not end)")

    # -------------------------------------------------------------- public API
    def health(self) -> bool:
        try:
            return self._http.get("/v1/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def products(self, category: str | None = None, active_only: bool = True) -> list[Product]:
        items = list(self._pages("/v1/products", Product, {"category": category}))
        return [p for p in items if p.active or not active_only]

    def followups(self, status: str | None = None, category: str | None = None,
                  client_ref: int | None = None) -> list[FollowUp]:
        return list(self._pages("/v1/followups", FollowUp,
                                {"status": status, "category": category, "client_ref": client_ref}))

    def close(self) -> None:
        self._http.close()
