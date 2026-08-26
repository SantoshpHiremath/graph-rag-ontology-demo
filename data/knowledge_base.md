# Internal Engineering Knowledge Base — Reference Notes

Synthetic internal documentation for a small e-commerce platform, used
as the source text for both vector-based RAG retrieval and graph-based
retrieval in this project. Not real company data.

## Services and ownership

- checkout-api is owned by Team Orion.
- payments-gateway is owned by Team Nimbus.
- inventory-service is owned by Team Orion.
- notification-service is owned by Team Vega.
- auth-service is owned by Team Nimbus.

## Service dependencies

- checkout-api depends on payments-gateway because it charges the
  customer's card during checkout.
- checkout-api depends on inventory-service because it reserves stock
  before confirming an order.
- checkout-api depends on auth-service because it validates the
  customer's session token.
- payments-gateway depends on notification-service because it sends a
  payment-confirmation email after a successful charge.
- inventory-service depends on notification-service because it sends a
  low-stock alert when reserved stock falls below a threshold.

## Incident history

- INC-101: Checkout errors during a payments-gateway outage. Caused by
  payments-gateway. Affected checkout-api. Resolved by Team Nimbus.
- INC-102: Stock oversell due to a race condition in inventory-service.
  Caused by inventory-service. Affected checkout-api. Resolved by Team
  Orion.
- INC-103: Missing order-confirmation emails after a
  notification-service deploy. Caused by notification-service. Affected
  payments-gateway. Resolved by Team Vega.
