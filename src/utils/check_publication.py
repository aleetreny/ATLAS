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
    python src/utils/check_publication.py            # comprueba, 0 si todo bien
    python src/utils/check_publication.py --next     # qué número y qué acentos quedan libres
    python src/utils/check_publication.py --reparte 3  # reparto para N sesiones a la vez

`--next` sirve para una sesión sola. Con varias a la vez **no vale**, y ese es el
agujero que dejó el choque original: tres sesiones que arrancan el mismo día leen
las tres el mismo "siguiente libre" y vuelven a chocar, solo que ahora el guardia
las caza en el push, con el artículo ya escrito. Para eso está `--reparte N`: lo
corre **quien lanza las sesiones**, una sola vez, y le da a cada una su número y
su acento ya asignados para pegarlos en el prompt. Un reparto hecho antes de
empezar no tiene carrera que perder.

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

# El secundario de los artículos publicados es el primario a un 80% por canal:
# medido contra cuatro parejas ya en el sitio, el peor canal se desvía 6 de 255
# (#7e22ce -> #6b1ba9 publicado, #651ba5 calculado) y tres de las cuatro caen
# dentro de 3. Se calcula en vez de escribirse a ojo para que un acento nuevo
# traiga su pareja sin que nadie tenga que pensarla.
SEC_FACTOR = 0.80


def secundario(primario):
    r, g, b = (int(primario[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c * SEC_FACTOR) for c in (r, g, b))


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

    libres = [c for c in RESERVA if c not in vistos]
    siguiente = max(nums) + 1 if nums else 1

    if "--next" in sys.argv:
        print(f"artículos publicados: {len(carpetas)}")
        print(f"siguiente número libre: {siguiente}")
        print(f"acentos libres de la reserva: {', '.join(libres) if libres else '(ninguno, elige uno nuevo)'}")
        print(f"acentos ya en uso: {len(vistos)}")

    if "--reparte" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--reparte") + 1])
        except (IndexError, ValueError):
            print("uso: --reparte N, con N el número de sesiones que vas a lanzar")
            return 1
        if n > len(libres):
            print(f"solo quedan {len(libres)} acentos en la reserva y pides {n}: "
                  f"añade colores a RESERVA antes de repartir")
            return 1
        print(f"\nreparto para {n} sesiones simultáneas, sobre {len(carpetas)} "
              f"artículos publicados.")
        print("pega en el prompt de cada sesión la línea que le toca, y ninguna "
              "tendrá que adivinar:\n")
        for i in range(n):
            col = libres[i]
            print(f"  sesión {i + 1}: tu artículo es el número {siguiente + i}, "
                  f"su acento es --primary: {col} y --secondary: {secundario(col)}. "
                  f"No los elijas tú ni los deduzcas del repositorio, ya están "
                  f"asignados y hay otras sesiones escribiendo a la vez.")
        print()

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
