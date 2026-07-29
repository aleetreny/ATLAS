"""Shared dataset access for the recommender articles (module 6.1).

A recommender needs the one thing none of the earlier modules had: a table
whose rows are people and whose columns are things, mostly empty. The public
set that is both real and readable is `goodbooks-10k`: six million one to five
star ratings from fifty three thousand readers over ten thousand books, with
titles and authors, which is what makes a recommendation something you can
judge by eye instead of by index.

Kept outside the repository like every other dataset here, in
`~/.atlas_vision_data`, and checked by digest so that a mirror quietly changing
under us shows up as a failure rather than as a different article.
"""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".atlas_vision_data"
BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
FILES = {
    "goodbooks_ratings.csv": ("ratings.csv", 72126826),
    "goodbooks_books.csv": ("books.csv", 3286659),
}
# Recorded on first download rather than pinned from elsewhere, so the check is
# "this file has not changed since the article was written", which is the
# property that matters.
DIGESTS = {
    "goodbooks_ratings.csv": "1ee8a97172bd6d97",
    "goodbooks_books.csv": "4d21d0e043312865",
}


def _fetch(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        remote, size = FILES[name]
        print(f"  downloading {remote} ({size / 1e6:.1f} MB) to {path}")
        with urllib.request.urlopen(BASE + remote, timeout=600) as r:
            path.write_bytes(r.read())
    return path


def digest(name, check=True):
    """El digest del fichero, y la comprobación de que sigue siendo el mismo.

    Un mirror que cambia debajo no da un error, da otro artículo: las cifras se
    mueven y nada avisa. Aquí se convierte en una excepción."""
    got = hashlib.sha256(_fetch(name).read_bytes()).hexdigest()[:16]
    if check and got != DIGESTS[name]:
        raise RuntimeError(f"{name} ha cambiado: {got} en vez de {DIGESTS[name]}")
    return got


def ratings():
    """(user, book, rating) as three int arrays, zero based ids.

    The file ships as `user_id,book_id,rating` with one based ids and no
    missing values; the only work here is turning three columns of text into
    three arrays without pandas."""
    path = _fetch("goodbooks_ratings.csv")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    return raw[:, 0] - 1, raw[:, 1] - 1, raw[:, 2].astype(np.int8)


def books():
    """Title and author per zero based book id, as two lists.

    The csv has quoted fields with commas inside them, so this is a small
    reader rather than a split on comma: the columns wanted are `book_id`,
    `authors`, `original_publication_year` and `title`."""
    import csv

    path = _fetch("goodbooks_books.csv")
    titles, authors, years = {}, {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            i = int(row["book_id"]) - 1
            titles[i] = row["title"]
            authors[i] = row["authors"]
            try:
                years[i] = int(float(row["original_publication_year"]))
            except (TypeError, ValueError):
                years[i] = None
    n = max(titles) + 1
    return ([titles.get(i, f"book {i}") for i in range(n)],
            [authors.get(i, "") for i in range(n)],
            [years.get(i) for i in range(n)])


def short_title(t, limit=42):
    """The titles carry their series in brackets: `Harry Potter and the
    Philosopher's Stone (Harry Potter, #1)`. A widget label has no room for
    that, and cutting at the bracket keeps the part a reader recognises."""
    t = t.split(" (")[0].strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"
