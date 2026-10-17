"""CRM integration tests against the mock API (in-process, synthetic data)."""
import time

import pytest
from fastapi.testclient import TestClient

from rse_agent import crm_client as cc
from rse_agent.mock_crm import FOLLOWUPS, PRODUCTS, MockConfig, create_app


def client(**cfg):
    conf = MockConfig(token="t0k", **cfg)
    http = TestClient(create_app(conf))
    return cc.CRMClient(base_url="http://testserver", token="t0k", http=http, backoff=0.01, max_retries=3)


def test_pagination_collects_every_page():
    c = client()
    items = list(c._pages("/v1/products", cc.Product, page_size=5))
    assert len(items) == len(PRODUCTS) and c.stats["pages"] == -(-len(PRODUCTS) // 5)


def test_filters_and_models():
    c = client()
    health = c.products(category="health_cover")
    assert health and all(isinstance(p, cc.Product) and p.category == "health_cover" and p.active for p in health)
    open_fu = c.followups(status="open")
    assert len(open_fu) == sum(f["status"] == "open" for f in FOLLOWUPS)


def test_bad_token_raises_auth_error_without_retrying():
    conf = MockConfig(token="right")
    c = cc.CRMClient(base_url="http://testserver", token="wrong", http=TestClient(create_app(conf)), backoff=0.01)
    with pytest.raises(cc.CRMAuthError):
        c.products()
    assert c.stats["retries"] == 0


def test_retries_through_intermittent_503():
    c = client(fail_every=2)          # every 2nd request fails, mid-pagination
    items = list(c._pages("/v1/products", cc.Product, page_size=5))
    assert len(items) == len(PRODUCTS)          # nothing lost or duplicated across retries
    assert c.stats["retries"] >= 1


def test_gives_up_with_clear_message_when_always_failing():
    c = client(fail_every=1)
    with pytest.raises(cc.CRMUnavailable, match="after 4 attempts"):
        c.products()


def test_rate_limit_honours_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr(cc.time, "sleep", lambda s: slept.append(s))
    c = client(rate_limit=1, rate_window=1)
    c.products()                       # first page ok
    with pytest.raises(cc.CRMUnavailable, match="rate limited"):
        c.followups()                  # immediately limited; sleep is mocked so the window never passes
    assert slept and all(s >= 1 for s in slept)


def test_schema_drift_becomes_validation_error():
    c = client(malformed=True)
    with pytest.raises(cc.CRMValidationError, match="unexpected data"):
        c.products()


def test_timeout_is_retried_then_reported(monkeypatch):
    import httpx

    class Slow(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ReadTimeout("slow", request=request)

    c = cc.CRMClient(base_url="http://x", token="t", http=httpx.Client(base_url="http://x", transport=Slow()),
                     backoff=0.01, max_retries=2, timeout=0.5)
    t0 = time.perf_counter()
    with pytest.raises(cc.CRMUnavailable, match="timed out"):
        c.products()
    assert c.stats["requests"] == 3 and time.perf_counter() - t0 < 5
