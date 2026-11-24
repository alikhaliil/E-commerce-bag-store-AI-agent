"""Seed the SQLite database with realistic sample data for the bag store.

Run from the project root:

    python -m app.db.seed

The script is idempotent: it clears ``products`` and ``orders`` before
inserting, so running it repeatedly always produces the same catalogue and
order set. ``support_tickets`` is left untouched because tickets are created
by the agent at runtime.

Edge cases deliberately included for later testing:
  * Out-of-stock products (stock = 0)
  * A low-stock product (stock = 2)
  * Orders with status 'delayed' (with and without a tracking number)
  * A 'cancelled' order and a 'pending' order with no tracking number
  * One customer with multiple orders (lookup by name returns >1 row)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.db.database import DB_PATH, get_db_connection, init_db

logger = logging.getLogger(__name__)

# (name, category, price, stock, description)
PRODUCTS: list[tuple[str, str, float, int, str]] = [
    (
        "Urban Commuter Backpack",
        "Backpacks",
        1450.00,
        34,
        "25L water-resistant backpack with a padded 15.6\" laptop sleeve, "
        "hidden anti-theft pocket and USB charging port.",
    ),
    (
        "Trailblazer Hiking Backpack",
        "Backpacks",
        2100.00,
        0,  # EDGE CASE: out of stock
        "40L hiking pack with adjustable torso length, rain cover and "
        "hydration bladder compartment.",
    ),
    (
        "Classic Leather Tote",
        "Totes",
        2750.00,
        12,
        "Full-grain leather tote with a zip-top closure, inner organiser "
        "pockets and reinforced handles.",
    ),
    (
        "Everyday Canvas Tote",
        "Totes",
        480.00,
        85,
        "Heavy 16oz cotton canvas tote with an inner zip pocket. Machine washable.",
    ),
    (
        "Mini Crossbody Bag",
        "Crossbody",
        890.00,
        2,  # EDGE CASE: low stock
        "Compact vegan-leather crossbody with adjustable strap and RFID-blocking "
        "card slots.",
    ),
    (
        "Executive Laptop Briefcase",
        "Laptop Bags",
        3200.00,
        9,
        "Slim leather briefcase for laptops up to 16\", with a trolley sleeve and "
        "padded tablet compartment.",
    ),
    (
        "Weekender Duffel Bag",
        "Travel",
        1950.00,
        0,  # EDGE CASE: out of stock
        "45L waxed-canvas duffel with a separate shoe compartment and detachable "
        "shoulder strap.",
    ),
    (
        "Cabin Carry-On Suitcase",
        "Travel",
        4100.00,
        15,
        "55cm hard-shell carry-on with 360° spinner wheels and a TSA-approved lock.",
    ),
    (
        "Kids' Dino School Bag",
        "Kids",
        650.00,
        40,
        "Lightweight school bag with ergonomic straps, reflective strips and a "
        "fun dinosaur print.",
    ),
    (
        "Evening Clutch",
        "Clutches",
        720.00,
        18,
        "Satin evening clutch with a magnetic clasp and detachable chain strap.",
    ),
    (
        "Waterproof Gym Sling",
        "Sports",
        560.00,
        27,
        "Roll-top waterproof sling bag with a ventilated wet/dry compartment.",
    ),
]


def _iso(days_ago: float) -> str:
    """UTC timestamp `days_ago` days before now, in ISO-8601 'Z' format."""
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_orders() -> list[tuple[str, str, str, str, str | None, float, str]]:
    """Return orders as (order_id, customer_name, product_name, status,
    tracking_number, total_price, updated_at).

    Timestamps are relative to "now" so 'delayed' and 'shipped' orders always
    look recent, whenever the seed is run.
    """
    return [
        ("ORD-1001", "Ahmed Hassan", "Urban Commuter Backpack", "delivered",
         "EGX123456789", 1450.00, _iso(12)),
        ("ORD-1002", "Mona Khaled", "Classic Leather Tote", "shipped",
         "EGX223344556", 2750.00, _iso(2)),
        ("ORD-1003", "Omar Farouk", "Executive Laptop Briefcase", "processing",
         None, 3200.00, _iso(0.5)),
        # EDGE CASE: delayed, already has a tracking number (stuck in transit)
        ("ORD-1004", "Sara Mahmoud", "Cabin Carry-On Suitcase", "delayed",
         "EGX998877665", 4100.00, _iso(6)),
        # EDGE CASE: delayed before dispatch (no tracking number yet)
        ("ORD-1005", "Youssef Adel", "Weekender Duffel Bag", "delayed",
         None, 1950.00, _iso(9)),
        ("ORD-1006", "Nour El-Din", "Mini Crossbody Bag", "pending",
         None, 890.00, _iso(0.1)),
        ("ORD-1007", "Ahmed Hassan", "Everyday Canvas Tote", "shipped",
         "EGX556677889", 960.00, _iso(1)),  # qty 2; repeat customer
        ("ORD-1008", "Laila Ibrahim", "Evening Clutch", "cancelled",
         None, 720.00, _iso(4)),
        ("ORD-1009", "Karim Mostafa", "Trailblazer Hiking Backpack", "delivered",
         "EGX101202303", 2100.00, _iso(20)),
        ("ORD-1010", "Hana Samir", "Kids' Dino School Bag", "returned",
         "EGX404505606", 650.00, _iso(15)),
    ]


def _validate(orders: list[tuple]) -> None:
    """Fail fast if an order references a product that is not in the catalogue."""
    product_names = {p[0] for p in PRODUCTS}
    unknown = {o[2] for o in orders} - product_names
    if unknown:
        raise ValueError(f"Orders reference unknown products: {sorted(unknown)}")


def seed() -> None:
    """Initialise the schema and (re)load products and orders."""
    init_db()
    orders = _build_orders()
    _validate(orders)

    # Single transaction: either everything is replaced or nothing is.
    with get_db_connection() as conn:
        conn.execute("DELETE FROM orders;")
        conn.execute("DELETE FROM products;")
        # Reset AUTOINCREMENT so product ids are stable (1..N) across runs.
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'products';")

        conn.executemany(
            """
            INSERT INTO products (name, category, price, stock, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            PRODUCTS,
        )
        conn.executemany(
            """
            INSERT INTO orders (order_id, customer_name, product_name, status,
                                tracking_number, total_price, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            orders,
        )

        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        out_of_stock = conn.execute(
            "SELECT COUNT(*) FROM products WHERE stock = 0"
        ).fetchone()[0]
        order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        delayed = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE status = 'delayed'"
        ).fetchone()[0]

    logger.info(
        "Seeded %s: %d products (%d out of stock), %d orders (%d delayed)",
        DB_PATH, product_count, out_of_stock, order_count, delayed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    seed()
