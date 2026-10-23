import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "shop.db"
 
SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    city        TEXT,
    joined_date TEXT NOT NULL DEFAULT (date('now'))
);
 
CREATE TABLE IF NOT EXISTS products (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    brand      TEXT NOT NULL,
    category   TEXT NOT NULL CHECK (category IN ('bike', 'accessory', 'parts')),
    model_year INTEGER,
    price      REAL NOT NULL CHECK (price >= 0),
    stock      INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0)
);
 
CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT NOT NULL DEFAULT (date('now')),
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'shipped', 'delivered', 'returned'))
);
 
CREATE TABLE IF NOT EXISTS order_items (
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price REAL NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
"""
 
DROP = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
"""

def init_db(reset: bool = False) -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        if reset:
            conn.executescript(DROP)
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    print(f"{'Reset' if reset else 'Initialised'} {DB_PATH}")
 
 
if __name__ == "__main__":
    init_db(reset="--reset" in sys.argv)
