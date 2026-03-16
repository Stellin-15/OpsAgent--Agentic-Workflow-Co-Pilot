"""Stripe billing client (Phase 6).

Handles:
    1. Checkout session creation (new team subscribes)
    2. Webhook event processing (payment succeeded → upgrade tier)
    3. Subscription management (cancel, upgrade, downgrade)

Pricing tiers map to Stripe Price IDs set in environment variables:
    STRIPE_PRICE_STARTER   — $29/month
    STRIPE_PRICE_TEAM      — $79/month
    STRIPE_PRICE_ENTERPRISE — $299/month

Usage:
    client = StripeClient(settings)
    url = await client.create_checkout_session(team_id, tier="starter")
    # Redirect user to url

Webhook endpoint: POST /api/billing/webhook
    Stripe signs the payload — we verify with STRIPE_WEBHOOK_SECRET.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# Maps tier name → metadata key for Stripe
TIER_TO_PRICE_ENV = {
    "starter": "STRIPE_PRICE_STARTER",
    "team": "STRIPE_PRICE_TEAM",
    "enterprise": "STRIPE_PRICE_ENTERPRISE",
}


class StripeClient:
    def __init__(self, settings) -> None:
        self._settings = settings
        self._stripe = None

    def _get_stripe(self):
        if self._stripe is None:
            import stripe
            stripe.api_key = self._settings.stripe_secret_key
            self._stripe = stripe
        return self._stripe

    def is_configured(self) -> bool:
        return bool(getattr(self._settings, "stripe_secret_key", ""))

    async def create_checkout_session(
        self,
        team_id: str,
        tier: str,
        success_url: str = "https://opsagent.io/billing/success",
        cancel_url: str = "https://opsagent.io/billing/cancel",
    ) -> str:
        """
        Create a Stripe Checkout session for a team upgrading to *tier*.
        Returns the hosted checkout URL to redirect the user to.
        """
        import os
        import asyncio

        stripe = self._get_stripe()
        price_id = os.getenv(TIER_TO_PRICE_ENV[tier], "")
        if not price_id:
            raise ValueError(f"No Stripe Price ID configured for tier '{tier}'")

        session = await asyncio.to_thread(
            stripe.checkout.Session.create,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            metadata={"team_id": team_id, "tier": tier},
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
        )
        log.info(
            "stripe.checkout_created",
            team_id=team_id,
            tier=tier,
            session_id=session.id,
        )
        return session.url

    async def handle_webhook(self, payload: bytes, signature: str) -> dict | None:
        """
        Verify and process a Stripe webhook event.
        Returns the processed event dict or None for unhandled events.
        """
        import asyncio

        stripe = self._get_stripe()
        webhook_secret = getattr(self._settings, "stripe_webhook_secret", "")

        try:
            event = await asyncio.to_thread(
                stripe.Webhook.construct_event,
                payload,
                signature,
                webhook_secret,
            )
        except Exception as exc:
            log.warning("stripe.webhook_invalid", error=str(exc))
            return None

        event_type = event["type"]
        log.info("stripe.webhook_received", event_type=event_type)

        if event_type == "checkout.session.completed":
            session = event["data"]["object"]
            team_id = session["metadata"].get("team_id")
            tier = session["metadata"].get("tier")
            if team_id and tier:
                log.info("stripe.subscription_activated", team_id=team_id, tier=tier)
                return {"team_id": team_id, "tier": tier, "action": "upgrade"}

        elif event_type in ("customer.subscription.deleted", "invoice.payment_failed"):
            # Downgrade to free tier
            subscription = event["data"]["object"]
            customer_id = subscription.get("customer")
            log.info(
                "stripe.subscription_cancelled",
                customer_id=customer_id,
                event=event_type,
            )
            return {"customer_id": customer_id, "action": "downgrade"}

        return None
