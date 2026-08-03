"""
DynamoDB trade reader.

Isolated from the sync command so it can be mocked in tests without
touching AWS, and so the paginated-scan and item-parsing concerns live in
one place. Returns plain dataclasses; the command decides how to persist.

The table is single-table (holdings, watchlists, cash, snapshots, trades
share it), so every read filters to entityType = TRADE. A trade item looks
like:

    userId:     <cognito sub>            (partition key)
    type:       trade#<iso8601>#<uuid>   (sort key)
    entityType: "TRADE"
    tradeId, symbol, side, quantity, price, executedAt, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Attr


@dataclass(frozen=True)
class TradeItem:
    user_sub: str
    trade_id: str
    symbol: str
    side: str            # "buy" / "sell", normalised lower
    quantity: Decimal
    price: Decimal
    executed_at: datetime


def _parse(item: dict) -> TradeItem | None:
    """
    Turn a raw DynamoDB item into a TradeItem, or None if it is malformed.

    Returning None rather than raising means one corrupt row doesn't abort
    a whole-table sync -- the command counts and logs skips instead.
    """
    try:
        return TradeItem(
            user_sub=item["userId"],
            trade_id=item["tradeId"],
            symbol=item["symbol"],
            side=str(item["side"]).lower(),
            quantity=Decimal(str(item["quantity"])),
            price=Decimal(str(item["price"])),
            executed_at=datetime.fromisoformat(
                item["executedAt"].replace("Z", "+00:00")
            ),
        )
    except (KeyError, ValueError, TypeError):
        return None


class TradeReader:
    def __init__(self, table_name: str, *, resource=None):
        self.table_name = table_name
        self._resource = resource or boto3.resource("dynamodb")
        self.table = self._resource.Table(table_name)

    def all_trades(self):
        """
        Yield every trade item in the table, following pagination.

        A Scan with a FilterExpression is the right tool for a full,
        cross-user batch: the filter is applied server-side so only TRADE
        items come back, and DynamoDB paginates automatically via
        LastEvaluatedKey. A Scan reads the whole table, which is fine for a
        periodic batch on a paper-trading dataset -- it is emphatically not
        something to do on a hot path.
        """
        scan_kwargs = {"FilterExpression": Attr("entityType").eq("TRADE")}

        while True:
            response = self.table.scan(**scan_kwargs)
            for raw in response.get("Items", []):
                parsed = _parse(raw)
                if parsed is not None:
                    yield parsed

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key