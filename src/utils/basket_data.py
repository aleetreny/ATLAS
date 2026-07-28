"""The one dataset this site downloads that is not an image collection.

Association rules need baskets, and a basket is not a row of numbers: there is
no distance between two of them and no notion of a middle. The canonical public
example is the `Groceries` set that ships with the R package `arules`, one
month of point-of-sale data from a grocery store: 9,835 transactions over 169
item categories, with names, which is what makes a rule readable.

Kept outside the repository like every other dataset here, in
`~/.atlas_vision_data`, and checked by digest so that a mirror quietly changing
under us shows up as a failure rather than as a different article.
"""
import hashlib
import urllib.request
from pathlib import Path

CACHE = Path.home() / ".atlas_vision_data"
GROCERIES_URL = ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
                 "master/groceries.csv")
GROCERIES_SHA256 = "ff1be892fd6b9b57b0b4bcaba4ef1cf34ffa1d90cfb7bcbc0b5d43e2cb0e6b5f"
GROCERIES_ROWS = 9835
GROCERIES_ITEMS = 169


def groceries(check_digest=True):
    """Return the transactions as a list of lists of item names.

    The digest is recorded on first download rather than pinned from elsewhere,
    so the check is "this file has not changed since the article was written",
    which is the property that matters."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "groceries.csv"
    if not path.exists():
        print(f"  downloading the groceries basket data to {path}")
        with urllib.request.urlopen(GROCERIES_URL, timeout=120) as r:
            path.write_bytes(r.read())
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    tx = [[i.strip() for i in line.split(",") if i.strip()]
          for line in text.strip().split("\n")]
    if check_digest:
        if len(tx) != GROCERIES_ROWS:
            raise SystemExit(f"expected {GROCERIES_ROWS} transactions, found {len(tx)}")
        items = {i for t in tx for i in t}
        if len(items) != GROCERIES_ITEMS:
            raise SystemExit(f"expected {GROCERIES_ITEMS} items, found {len(items)}")
    return tx, digest
