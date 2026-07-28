"""El guardia de publicación: lo que dos sesiones a la vez rompen sin querer.

Nace de un choque real. Tres sesiones en la nube escribieron un artículo cada
una el mismo día, en tres ramas, y las tres numeraron el suyo **27** y dos
eligieron el mismo acento (`#7e22ce`). Nada de eso lo detecta git: los ficheros
que chocan de verdad (`docs/estado.md`, `index.html`) dan conflicto de texto y
se resuelven a mano, y una resolución a mano puede dejar dos artículos con el
mismo número, un chip apuntando a una carpeta que no existe, o una miniatura
que nadie generó. El trabajo no se pierde nunca (git no borra ramas), pero la
portada queda mintiendo y eso sí se publica.

Comprueba, sobre el árbol de trabajo:

  1. cada artículo tiene un número, único y sin huecos, en la tabla de estado
  2. cada fila de la tabla apunta a una carpeta que existe, y cada carpeta de
     artículo tiene su fila
  3. cada artículo tiene un acento propio, y el que dice la tabla es el que
     está de verdad en su `index.html`
  4. cada artículo tiene tarjeta en la portada, con miniatura que existe
  5. cada chip vivo apunta a una carpeta que existe, y ninguna carpeta
     publicada se queda sin chip
  6. no hay rayas largas ni medias en nada de lo que se lee

Uso:
    python src/utils/check_publication.py          # comprueba, 0 si todo bien
    python src/utils/check_publication.py --next   # qué número y qué acentos quedan libres

Sale con código 1 si algo falla, para que sirva en CI y en un hook.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ESTADO = ROOT / "docs" / "estado.md"
PORTADA = ROOT / "index.html"

# Los acentos del sitio son oscuros a propósito: el texto va en blanco encima.
# Esta reserva es de dónde tirar cuando toque uno nuevo, no una regla.
RESERVA = [
    "#1d4ed8", "#3f6212", "#7c2d12", "#134e4a", "#581c87", "#831843",
    "#164e63", "#713f12", "#3730a3", "#065f46", "#701a75", "#7f1d1d",
]


def articulos_en_disco():
    """Una carpeta es un artículo si tiene index.html y js/main.js."""
    out = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        if (d / "index.html").is_file() and (d / "js" / "main.js").is_file():
            out.append(d.name)
    return out


def acento(carpeta):
    t = (ROOT / carpeta / "index.html").read_text("utf-8")
    m = re.search(r"--primary:\s*(#[0-9a-fA-F]{6})", t)
    return m.group(1).lower() if m else None


def filas_de_estado():
    """(numero, titulo, carpeta, color) por cada fila de la tabla."""
    filas = []
    for line in ESTADO.read_text("utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|(.+?)\|\s*`([^`]+)/`\s*\|\s*`(#[0-9a-fA-F]{6})`\s*\|",
                     line)
        if m:
            filas.append((int(m.group(1)), m.group(2).strip(), m.group(3),
                          m.group(4).lower()))
    return filas


def portada():
    t = PORTADA.read_text("utf-8")
    tarjetas = {}
    for m in re.finditer(
            r'<div class="imgBx" onclick="location\.href=\'\./([^\']+)/\';".*?'
            r'<img src="([^"]+)"', t, re.S):
        tarjetas[m.group(1)] = m.group(2)
    chips = []
    for m in re.finditer(r'<span class="chip live"><a href="\./([^"]+)/">', t):
        chips.append(m.group(1))
    return tarjetas, chips


def sin_rayas():
    """La regla dura 2 de CLAUDE.md, comprobada donde se lee.

    Los dos caracteres se construyen por punto de código y no se escriben aquí,
    para que este fichero no se acuse a sí mismo. Y una línea que habla de la
    regla necesita poder citarlos: se salta la que diga "rayas largas" o lleve
    el token `no-raya-check`.
    """
    EM, EN = chr(0x2014), chr(0x2013)
    malos = []
    patrones = ["*.html", "*.js", "*.py", "*.md", "*.css"]
    saltar = ("assets/js/vendor", "node_modules", ".git")
    for pat in patrones:
        for f in ROOT.rglob(pat):
            rel = f.relative_to(ROOT).as_posix()
            if rel.startswith(saltar):
                continue
            try:
                t = f.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(t.splitlines(), 1):
                if EM not in line and EN not in line:
                    continue
                if "rayas largas" in line or "no-raya-check" in line:
                    continue
                malos.append(f"{rel}:{i}")
    return malos


def main():
    fallos = []
    avisos = []

    carpetas = articulos_en_disco()
    filas = filas_de_estado()
    tarjetas, chips = portada()

    # ---- 1. números únicos y sin huecos
    nums = [n for n, _, _, _ in filas]
    repes = sorted({n for n in nums if nums.count(n) > 1})
    if repes:
        for n in repes:
            quienes = [c for num, _, c, _ in filas if num == n]
            fallos.append(f"el número {n} lo reclaman {len(quienes)} artículos: "
                          f"{', '.join(quienes)}")
    esperado = list(range(1, len(filas) + 1))
    if sorted(nums) != esperado and not repes:
        faltan = sorted(set(esperado) - set(nums))
        fallos.append(f"la numeración tiene huecos: falta {faltan}")

    # ---- 2. tabla contra disco
    en_tabla = {c for _, _, c, _ in filas}
    for _, _, c, _ in filas:
        if c not in carpetas:
            fallos.append(f"la tabla de estado.md cita `{c}/`, que no existe en disco")
    for c in carpetas:
        if c not in en_tabla:
            fallos.append(f"la carpeta `{c}/` no tiene fila en la tabla de estado.md")

    # ---- 3. un acento por artículo, y el que dice la tabla
    vistos = {}
    for c in carpetas:
        col = acento(c)
        if col is None:
            fallos.append(f"`{c}/index.html` no declara --primary")
            continue
        if col in vistos:
            fallos.append(f"el acento {col} lo usan dos artículos: {vistos[col]} y {c}")
        else:
            vistos[col] = c
    for _, _, c, col in filas:
        real = acento(c) if c in carpetas else None
        if real and real != col:
            fallos.append(f"estado.md dice que `{c}/` es {col} y su index.html dice {real}")

    # ---- 4. tarjeta y miniatura por artículo
    for c in carpetas:
        if c not in tarjetas:
            fallos.append(f"`{c}/` no tiene tarjeta en la portada")
            continue
        thumb = ROOT / tarjetas[c]
        if not thumb.is_file():
            fallos.append(f"la tarjeta de `{c}/` apunta a {tarjetas[c]}, que no existe")
    for c, src in tarjetas.items():
        if c not in carpetas:
            fallos.append(f"la portada tiene una tarjeta a `{c}/`, que no es un artículo")

    # ---- 5. chips
    for c in chips:
        if c not in carpetas:
            fallos.append(f"un chip vivo apunta a `{c}/`, que no existe")
    sin_chip = [c for c in carpetas if c not in chips]
    if sin_chip:
        avisos.append(f"publicados sin chip vivo en el mapa de ramas: {', '.join(sin_chip)}")

    # ---- 6. rayas
    malos = sin_rayas()
    if malos:
        fallos.append(f"rayas largas o medias en {len(malos)} sitios: "
                      f"{', '.join(malos[:6])}{' ...' if len(malos) > 6 else ''}")

    if "--next" in sys.argv:
        libres = [c for c in RESERVA if c not in vistos]
        print(f"artículos publicados: {len(carpetas)}")
        print(f"siguiente número libre: {max(nums) + 1 if nums else 1}")
        print(f"acentos libres de la reserva: {', '.join(libres) if libres else '(ninguno, elige uno nuevo)'}")
        print(f"acentos ya en uso: {len(vistos)}")

    for a in avisos:
        print(f"aviso: {a}")
    if fallos:
        print(f"\nFALLA con {len(fallos)} problema(s) de publicación:")
        for f in fallos:
            print(f"  - {f}")
        return 1
    print(f"\npublicación coherente: {len(carpetas)} artículos, "
          f"{len(vistos)} acentos distintos, {len(chips)} chips vivos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
