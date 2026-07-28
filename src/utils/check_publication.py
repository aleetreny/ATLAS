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
  7. el orden de las tarjetas es el orden de lectura, y cada artículo enlaza
     al siguiente
  8. los chips de cada rama van en ese mismo orden, agrupados por artículo

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
import subprocess
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


def git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def acentos_en_vuelo(ya_publicados):
    """Los acentos que las ramas sin mergear ya están usando, y nadie ve.

    Sin esto el reparto miente, y miente en la dirección que más duele: mira
    solo `main`, así que le da a una sesión nueva un color que otra rama lleva
    dos días pintando y que nadie ha publicado todavía. Es el mismo agujero de
    siempre visto desde otro lado, y aquí sí tiene arreglo, porque las ramas de
    las otras sesiones están en origin y se pueden leer.

    Devuelve {color: rama}. Si no hay git, o no hay refs remotas (en CI se
    clona una rama sola), devuelve vacío: esto informa al reparto, nunca
    decide si el guardia pasa o falla.
    """
    refs = git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
    if not refs:
        return {}
    fuera = {}
    for rama in refs.split():
        if rama in ("origin/HEAD", "origin/main"):
            continue
        if git("merge-base", "--is-ancestor", rama, "origin/main") is not None:
            continue                       # ya está en main, sus colores ya se ven
        arbol = git("ls-tree", "-r", "--name-only", rama)
        if not arbol:
            continue
        for path in arbol.split():
            m = re.fullmatch(r"([^/]+)/index\.html", path)
            # Solo las carpetas que la rama trae NUEVAS: una rama arrastra los
            # treinta artículos ya publicados y listarlos no informa de nada.
            if not m or m.group(1) in ya_publicados:
                continue
            t = git("show", f"{rama}:{path}")
            c = re.search(r"--primary:\s*(#[0-9a-fA-F]{6})", t or "")
            if c:
                fuera.setdefault(c.group(1).lower(), f"{rama} ({m.group(1)}/)")
    return fuera


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


def ramas():
    """Los chips vivos de cada rama del mapa, en el orden en que se leen.

    Devuelve [(titulo, [carpeta, ...]), ...]. Si la portada no declara ramas
    (el sitio de prueba no lo hace), todos los chips vivos cuentan como una.
    """
    t = PORTADA.read_text("utf-8")
    fuera = []
    for m in re.finditer(r'<h4>(.*?)</h4>(.*?)</div>', t, re.S):
        vivos = re.findall(r'<span class="chip live"><a href="\./([^"]+)/">', m.group(2))
        if vivos:
            fuera.append((re.sub(r'<[^>]+>|&[a-z]+;', ' ', m.group(1)).strip(), vivos))
    if not fuera:
        _, chips = portada()
        fuera = [("el mapa entero", chips)] if chips else []
    return fuera


def enlaza_a(carpeta, destino):
    """¿La prosa de `carpeta` manda al lector a `destino`?

    Se mira el HTML y su `prose.js`, que son los dos sitios donde vive prosa:
    a partir del artículo 21 los cierres se componen desde el JSON al cargar,
    así que el enlace del cierre puede no estar escrito en el HTML.
    """
    fuentes = [ROOT / carpeta / "index.html", ROOT / carpeta / "js" / "prose.js"]
    aguja = f"../{destino}/"
    return any(f.is_file() and aguja in f.read_text("utf-8") for f in fuentes)


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

    # ---- 7. la cadena de lectura, que es el orden de las tarjetas
    #
    # Tres sesiones en paralelo escribieron tres secciones y ninguna podía ver
    # los cierres de las otras dos: una dejó su "next article in this branch"
    # apuntando a la portada esperando una continuación que ya existía, y otra
    # mandó al lector a mitad de otro módulo. Ni git ni un comprobador de
    # enlaces rotos ven nada, porque `../` existe y la otra carpeta también.
    # Lo que sí lo ve es preguntarle a la portada quién va después y buscar ese
    # enlace en la página de antes.
    lectura = [c for c in tarjetas if c in carpetas]
    for antes, despues in zip(lectura, lectura[1:]):
        if not enlaza_a(antes, despues):
            fallos.append(f"la cadena de lectura se corta: `{antes}/` no enlaza a "
                          f"`{despues}/`, que es la tarjeta siguiente en la portada")

    # ---- 8. los chips de cada rama, en el orden de las tarjetas
    #
    # Un chip es un algoritmo y una tarjeta es un artículo, así que un artículo
    # enciende varios chips seguidos; lo que no puede pasar es que la rama los
    # baraje, porque entonces el mapa cuenta un orden de lectura y la galería
    # cuenta otro. El orden bueno es el taxonómico de las tarjetas, no el
    # cronológico en el que se fueron escribiendo.
    posicion = {c: i for i, c in enumerate(lectura)}
    for titulo, vivos in ramas():
        idx = [posicion[c] for c in vivos if c in posicion]
        if idx != sorted(idx):
            fallos.append(f"los chips de «{titulo}» no siguen el orden de las tarjetas: "
                          f"{', '.join(vivos)}")

    quiere_reparto = "--reparte" in sys.argv
    en_vuelo = (acentos_en_vuelo(set(carpetas))
                if (quiere_reparto or "--next" in sys.argv) else {})
    ocupados = set(vistos) | set(en_vuelo)
    libres = [c for c in RESERVA if c not in ocupados]
    siguiente = max(nums) + 1 if nums else 1

    if "--next" in sys.argv:
        print(f"artículos publicados: {len(carpetas)}")
        print(f"siguiente número libre: {siguiente}")
        print(f"acentos libres de la reserva: {', '.join(libres) if libres else '(ninguno, elige uno nuevo)'}")
        print(f"acentos en uso: {len(vistos)} publicados"
              + (f" y {len(en_vuelo)} en ramas sin mergear" if en_vuelo else ""))
        for col, quien in sorted(en_vuelo.items()):
            print(f"    {col} lo está usando {quien}")

    if quiere_reparto:
        try:
            n = int(sys.argv[sys.argv.index("--reparte") + 1])
        except (IndexError, ValueError):
            print("uso: --reparte N, con N el número de sesiones que vas a lanzar")
            return 1
        if n > len(libres):
            print(f"solo quedan {len(libres)} acentos sin pedir y necesitas {n}: "
                  f"añade colores a RESERVA antes de repartir")
            return 1
        print(f"\nreparto para {n} sesiones simultáneas, sobre {len(carpetas)} "
              f"artículos publicados"
              + (f" y {len(en_vuelo)} acento(s) ya pedidos por ramas en vuelo" if en_vuelo else "")
              + ".")
        print("pega en el prompt de cada sesión la línea que le toca:\n")
        for i in range(n):
            col = libres[i]
            print(f"  sesión {i + 1}: el acento de tus artículos es "
                  f"--primary: {col} y --secondary: {secundario(col)}. No lo elijas "
                  f"tú ni lo deduzcas del repositorio: hay otras sesiones "
                  f"escribiendo a la vez y el reparto ya está hecho. El NÚMERO de "
                  f"cada artículo no se reparte: lo coges al mergear, con "
                  f"--next, que es cuando ya no hay carrera.")
        print("\n  (si una sesión escribe varios artículos, el segundo y los "
              "siguientes llevan\n   variantes del mismo acento o colores nuevos "
              "que --next diga libres en ese momento)")
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
