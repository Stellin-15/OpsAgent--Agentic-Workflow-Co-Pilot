"""Stripe billing API endpoints (Phase 6).

POST /api/billing/checkout/{tier}  — create Stripe checkout session
POST /api/billing/webhook          — receive Stripe webhook events
GET  /api/billing/status           — current team subscription status
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from opsagent.auth.jwt import TokenPayload, require_auth
from opsagent.billing.stripe_client import StripeClient
from opsagent.config import get_settings

log = structlog.get_logger(__name__)
router = APIRouter(tags=["billing"])


def _get_stripe() -> StripeClient:
    return StripeClient(get_settings())


@router.post("/billing/checkout/{tier}")
async def create_checkout(
    tier: str,
    token: TokenPayload = Depends(require_auth),
    stripe: StripeClient = Depends(_get_stripe),
):
    """Create a Stripe Checkout session for the authenticated team."""
    valid_tiers = {"starter", "team", "enterprise"}
    if tier not in valid_tiers:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Choose from {valid_tiers}")

    if not stripe.is_configured():
        raise HTTPException(status_code=501, detail="Billing not configured")

    url = await stripe.create_checkout_session(
        team_id=token.team_id,
        tier=tier,
    )
    return {"checkout_url": url}


@router.post("/billing/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe webhook receiver.

    Stripe signs every event — we verify the signature before processing.
    Register this URL in your Stripe Dashboard → Webhooks.
    """
    payload = await request.body()
    signature = request.headers.get("Stripe-Signature", "")
    settings = get_settings()
    stripe = StripeClient(settings)

    event = await stripe.handle_webhook(payload, signature)
    if event is None:
        return JSONResponse({"status": "ignored"})

    if event.get("action") == "upgrade":
        # In production: update the team's tier in the DB
        log.info("billing.team_upgraded", team_id=event["team_id"], tier=event["tier"])

    elif event.get("action") == "downgrade":
        log.info("billing.team_downgraded", customer_id=event.get("customer_id"))

    return JSONResponse({"status": "processed"})


@router.get("/billing/status")
async def billing_status(token: TokenPayload = Depends(require_auth)):
    """Return the current subscription tier for the authenticated team."""
    return {
        "team_id": token.team_id,
        "tier": token.tier,
        "expires_at": token.exp.isoformat(),
    }
