"""Mock CRM REST API (synthetic data) used to demonstrate real integration patterns.

    python -m rse_agent.mock_crm            # serves http://127.0.0.1:8765  (token: RSE_CRM_TOKEN)

Endpoints (all JSON, all except /v1/health need `Authorization: Bearer <token>`):
    GET /v1/health
    GET /v1/products?category=&page=&page_size=      product catalogue (fictional providers)
    GET /v1/products/{product_id}
    GET /v1/followups?status=&category=&client_ref=&page=&page_size=   RM follow-up tasks

Behaviours a real API has, so the client must handle them:
    - pagination: {"data": [...], "page", "page_size", "total", "next_page"}
    - authentication errors: 401 {"error": "invalid_token"}
    - rate limiting: 429 + Retry-After once more than RATE_LIMIT requests arrive in RATE_WINDOW s
    - simulated faults (env or MockConfig): every Nth request -> 503, every Mth -> slow, or a
      malformed payload (missing fields) to exercise JSON validation
All data is synthetic and generated from a fixed seed; client_ref values are row numbers only.
"""
from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8765
CATEGORIES = ["term_cover", "health_cover", "sip", "swp", "pms", "las", "li_premium", "ulip"]


@dataclass
class MockConfig:
    token: str = field(default_factory=lambda: os.getenv("RSE_CRM_TOKEN", "demo-token"))
    rate_limit: int = int(os.getenv("RSE_CRM_RATE_LIMIT", "30"))      # requests per window
    rate_window: float = float(os.getenv("RSE_CRM_RATE_WINDOW", "10"))
    fail_every: int = int(os.getenv("RSE_CRM_FAIL_EVERY", "0"))       # 0 = never return 503
    slow_every: int = int(os.getenv("RSE_CRM_SLOW_EVERY", "0"))       # 0 = never slow
    slow_seconds: float = float(os.getenv("RSE_CRM_SLOW_SECONDS", "3"))
    malformed: bool = os.getenv("RSE_CRM_MALFORMED", "0") == "1"


# ---------------------------------------------------------------- synthetic data
def make_products(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    providers = {"term_cover": ["Alpha Life", "Bharat Shield Life"], "health_cover": ["Nova Health", "CarePlus Health"],
                 "sip": ["Horizon MF", "Lotus MF"], "swp": ["Horizon MF", "Lotus MF"], "pms": ["Summit PMS"],
                 "las": ["Keystone Finance"], "li_premium": ["Alpha Life"], "ulip": ["Bharat Shield Life"]}
    names = {"term_cover": ["SecureLife Term", "FamilyGuard Term", "IncomeShield Term"],
             "health_cover": ["CareFirst Family Floater", "SuperTopUp Plus", "Senior Care Health"],
             "sip": ["Flexi Cap Fund", "Large & Mid Cap Fund", "Balanced Advantage Fund", "Index 50 Fund"],
             "swp": ["Retirement Income SWP", "Conservative Hybrid SWP"], "pms": ["Quality Growth PMS"],
             "las": ["Securities Credit Line"], "li_premium": ["Guaranteed Savings Plan", "Child Future Plan"],
             "ulip": ["Wealth Builder ULIP"]}
    tickets = {"term_cover": (8000, 30000), "health_cover": (12000, 45000), "sip": (500, 5000),
               "swp": (100000, 500000), "pms": (5000000, 5000000), "las": (100000, 200000),
               "li_premium": (24000, 100000), "ulip": (24000, 60000)}
    risk = {"term_cover": "low", "health_cover": "low", "sip": "medium", "swp": "medium", "pms": "high",
            "las": "high", "li_premium": "low", "ulip": "medium"}
    out, pid = [], 1
    for cat in CATEGORIES:
        for nm in names[cat]:
            lo, hi = tickets[cat]
            out.append({"product_id": f"P{pid:03d}", "name": nm, "category": cat,
                        "provider": rng.choice(providers[cat]),
                        "min_ticket_inr": int(round(rng.uniform(lo, hi), -2)),
                        "risk_level": risk[cat], "active": pid % 11 != 0})
            pid += 1
    return out


def make_followups(seed: int = 11, n_clients: int = 18) -> list[dict]:
    rng = random.Random(seed)
    start = date(2026, 10, 1)
    out = []
    for i in range(1, 33):
        cat = rng.choice(["term_cover", "health_cover", "health_cover", "sip", "swp", "li_premium", "term_cover"])
        out.append({"followup_id": f"F{i:03d}", "client_ref": rng.randint(1, n_clients), "product_category": cat,
                    "status": rng.choice(["open", "open", "done"]),
                    "due_date": (start + timedelta(days=rng.randint(0, 75))).isoformat(),
                    "owner": rng.choice(["RM-1", "RM-2", "RM-3"]),
                    "note": rng.choice(["Discuss cover gap", "Annual review", "Share proposal",
                                        "Collect documents", "Explain plan features"])})
    return out


PRODUCTS, FOLLOWUPS = make_products(), make_followups()


# ---------------------------------------------------------------- app
def create_app(config: MockConfig | None = None) -> FastAPI:
    cfg = config or MockConfig()
    app = FastAPI(title="Mock CRM API", version="1.0")
    hits: deque[float] = deque()
    counter = {"n": 0}
    lock = threading.Lock()

    @app.middleware("http")
    async def faults(request: Request, call_next):
        if request.url.path == "/v1/health":
            return await call_next(request)
        with lock:
            counter["n"] += 1
            n = counter["n"]
            now = time.monotonic()
            while hits and now - hits[0] > cfg.rate_window:
                hits.popleft()
            hits.append(now)
            limited = len(hits) > cfg.rate_limit
        if limited:
            retry = max(1, int(cfg.rate_window - (now - hits[0])) + 1)
            return JSONResponse({"error": "rate_limited", "detail": "too many requests"}, status_code=429,
                                headers={"Retry-After": str(retry)})
        if cfg.fail_every and n % cfg.fail_every == 0:
            return JSONResponse({"error": "service_unavailable"}, status_code=503)
        if cfg.slow_every and n % cfg.slow_every == 0:
            time.sleep(cfg.slow_seconds)
        return await call_next(request)

    def auth(authorization: str | None = Header(default=None)):
        if authorization != f"Bearer {cfg.token}":
            raise HTTPException(status_code=401, detail="invalid_token")

    def page_of(items: list[dict], page: int, page_size: int) -> dict:
        total = len(items)
        start = (page - 1) * page_size
        data = items[start:start + page_size]
        if cfg.malformed and data:
            data = [{k: v for k, v in d.items() if k not in ("name", "client_ref")} for d in data]
        return {"data": data, "page": page, "page_size": page_size, "total": total,
                "next_page": page + 1 if start + page_size < total else None}

    @app.exception_handler(HTTPException)
    async def http_error(_, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "products": len(PRODUCTS), "followups": len(FOLLOWUPS)}

    @app.get("/v1/products", dependencies=[Depends(auth)])
    def products(category: str | None = None, page: int = Query(1, ge=1),
                 page_size: int = Query(10, ge=1, le=50)):
        items = [p for p in PRODUCTS if category in (None, p["category"])]
        return page_of(items, page, page_size)

    @app.get("/v1/products/{product_id}", dependencies=[Depends(auth)])
    def product(product_id: str):
        for p in PRODUCTS:
            if p["product_id"] == product_id:
                return p
        raise HTTPException(status_code=404, detail="product_not_found")

    @app.get("/v1/followups", dependencies=[Depends(auth)])
    def followups(status: str | None = None, category: str | None = None, client_ref: int | None = None,
                  page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=50)):
        items = [f for f in FOLLOWUPS if status in (None, f["status"]) and category in (None, f["product_category"])
                 and client_ref in (None, f["client_ref"])]
        return page_of(items, page, page_size)

    return app


app = create_app()
_server_thread: threading.Thread | None = None


def ensure_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Start the mock CRM in a background thread if nothing answers on host:port (demo convenience)."""
    global _server_thread
    import httpx
    import uvicorn

    try:
        if httpx.get(f"http://{host}:{port}/v1/health", timeout=1).status_code == 200:
            return True
    except httpx.HTTPError:
        pass
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    _server_thread = threading.Thread(target=server.run, daemon=True)
    _server_thread.start()
    for _ in range(50):
        try:
            if httpx.get(f"http://{host}:{port}/v1/health", timeout=1).status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.1)
    return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
