"""Que el guardia de publicación pueda disparar, regla por regla.

Un guardia que calla solo sirve si se ha comprobado que sabe hablar: la trampa
está anotada en verificacion.md y ya costó un fallo real ("antes de creerse uno
que calla, comprobar que puede disparar"). Aquí se monta un sitio de mentira
mínimo en un directorio temporal, se rompe **una** cosa cada vez, y se exige que
el guardia falle y que el mensaje mencione lo que se rompió.

    python src/utils/test_check_publication.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve()
GUARDIA = AQUI.parent / "check_publication.py"

ESTADO = """# estado de mentira

| # | Artículo | Carpeta | Tema |
|---|---|---|---|
| 1 | Uno (algo) | `uno/` | `#111111` |
| 2 | Dos (algo) | `dos/` | `#222222` |
"""

PORTADA = """<!DOCTYPE html>
<html><body>
  <div class="article-card">
    <div class="imgBx" onclick="location.href='./uno/';" style="cursor: pointer">
      <img src="assets/thumbnails/uno.svg" alt="uno" />
    </div>
  </div>
  <div class="article-card">
    <div class="imgBx" onclick="location.href='./dos/';" style="cursor: pointer">
      <img src="assets/thumbnails/dos.svg" alt="dos" />
    </div>
  </div>
  <div class="chips">
    <span class="chip live"><a href="./uno/">Uno</a></span>
    <span class="chip live"><a href="./dos/">Dos</a></span>
  </div>
</body></html>
"""

ARTICULO = """<!DOCTYPE html>
<html><head><style>:root { --primary: %s; --secondary: #000000; }</style></head>
<body>%s</body></html>
"""

# La prosa de "uno" manda al lector a "dos", que es la tarjeta siguiente en la
# portada de mentira. Sin esa frase el sitio de control no pasa limpio, que es
# justo lo que la regla de la cadena de lectura exige.
CIERRE = '<p>Next: <a href="../dos/">Dos</a>.</p>'


def sitio(raiz):
    """El sitio mínimo que el guardia sabe leer, y que pasa limpio."""
    (raiz / "docs").mkdir(parents=True)
    (raiz / "docs" / "estado.md").write_text(ESTADO, encoding="utf-8")
    (raiz / "index.html").write_text(PORTADA, encoding="utf-8")
    (raiz / "assets" / "thumbnails").mkdir(parents=True)
    for nombre, color, cierre in [("uno", "#111111", CIERRE), ("dos", "#222222", "")]:
        (raiz / nombre / "js").mkdir(parents=True)
        (raiz / nombre / "index.html").write_text(ARTICULO % (color, cierre), encoding="utf-8")
        (raiz / nombre / "js" / "main.js").write_text("// nada\n", encoding="utf-8")
        (raiz / "assets" / "thumbnails" / f"{nombre}.svg").write_text("<svg/>", encoding="utf-8")
    dest = raiz / "src" / "utils"
    dest.mkdir(parents=True)
    shutil.copy(GUARDIA, dest / GUARDIA.name)
    return dest / GUARDIA.name


def corre(script):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def rompe_numero(raiz):
    p = raiz / "docs" / "estado.md"
    p.write_text(p.read_text("utf-8").replace("| 2 | Dos", "| 1 | Dos"), encoding="utf-8")
    return "el número 1 lo reclaman"


def rompe_acento(raiz):
    p = raiz / "dos" / "index.html"
    p.write_text(ARTICULO % ("#111111", CIERRE), encoding="utf-8")
    return "el acento #111111 lo usan dos"


def rompe_tabla_vs_css(raiz):
    p = raiz / "docs" / "estado.md"
    p.write_text(p.read_text("utf-8").replace("`#222222`", "`#333333`"), encoding="utf-8")
    return "y su index.html dice"


def rompe_carpeta_fantasma(raiz):
    p = raiz / "docs" / "estado.md"
    p.write_text(p.read_text("utf-8") + "| 3 | Tres (algo) | `tres/` | `#333333` |\n",
                 encoding="utf-8")
    return "que no existe en disco"


def rompe_articulo_sin_fila(raiz):
    (raiz / "tres" / "js").mkdir(parents=True)
    (raiz / "tres" / "index.html").write_text(ARTICULO % ("#333333", ""), encoding="utf-8")
    (raiz / "tres" / "js" / "main.js").write_text("// nada\n", encoding="utf-8")
    return "no tiene fila en la tabla"


def rompe_miniatura(raiz):
    (raiz / "assets" / "thumbnails" / "dos.svg").unlink()
    return "que no existe"


TARJETA_DOS = """  <div class="article-card">
    <div class="imgBx" onclick="location.href='./dos/';" style="cursor: pointer">
      <img src="assets/thumbnails/dos.svg" alt="dos" />
    </div>
  </div>
"""


def rompe_tarjeta(raiz):
    p = raiz / "index.html"
    t = p.read_text("utf-8")
    assert TARJETA_DOS in t, "la tarjeta tiene que existir antes de quitarla"
    p.write_text(t.replace(TARJETA_DOS, ""), encoding="utf-8")
    return "no tiene tarjeta en la portada"


def rompe_chip(raiz):
    p = raiz / "index.html"
    p.write_text(p.read_text("utf-8").replace('href="./dos/">Dos', 'href="./cuatro/">Cuatro'),
                 encoding="utf-8")
    return "apunta a `cuatro/`, que no existe"


def rompe_cadena(raiz):
    p = raiz / "uno" / "index.html"
    p.write_text(p.read_text("utf-8").replace('href="../dos/"', 'href="../"'), encoding="utf-8")
    return "la cadena de lectura se corta"


def rompe_orden_chips(raiz):
    p = raiz / "index.html"
    t = p.read_text("utf-8")
    uno = '<span class="chip live"><a href="./uno/">Uno</a></span>'
    dos = '<span class="chip live"><a href="./dos/">Dos</a></span>'
    assert uno in t and dos in t, "los dos chips tienen que existir antes de cruzarlos"
    p.write_text(t.replace(uno, "@@").replace(dos, uno).replace("@@", dos), encoding="utf-8")
    return "no siguen el orden de las tarjetas"


def rompe_raya(raiz):
    p = raiz / "uno" / "index.html"
    p.write_text(p.read_text("utf-8").replace("<body>", "<body>texto con " + chr(0x2014)),
                 encoding="utf-8")
    return "rayas largas o medias"


CASOS = [
    ("dos artículos con el mismo número", rompe_numero),
    ("dos artículos con el mismo acento", rompe_acento),
    ("la tabla y el css discrepan de color", rompe_tabla_vs_css),
    ("una fila apunta a una carpeta que no existe", rompe_carpeta_fantasma),
    ("un artículo sin fila en la tabla", rompe_articulo_sin_fila),
    ("una miniatura que falta", rompe_miniatura),
    ("un artículo sin tarjeta", rompe_tarjeta),
    ("un chip vivo a la nada", rompe_chip),
    ("un cierre que no lleva al siguiente", rompe_cadena),
    ("los chips de una rama barajados", rompe_orden_chips),
    ("una raya larga", rompe_raya),
]


def main():
    fallos = []

    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp) / "sitio"
        raiz.mkdir()
        script = sitio(raiz)
        code, salida = corre(script)
        if code != 0:
            print("el sitio de mentira no pasa limpio, así que no prueba nada:")
            print(salida)
            return 1
        print("ok  el sitio de control pasa limpio")

    for nombre, romper in CASOS:
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp) / "sitio"
            raiz.mkdir()
            script = sitio(raiz)
            esperado = romper(raiz)
            code, salida = corre(script)
            if code == 0:
                fallos.append(f"{nombre}: el guardia NO disparó")
                print(f"FALLA  {nombre}: no disparó")
            elif esperado not in salida:
                fallos.append(f"{nombre}: disparó por otra cosa")
                print(f"FALLA  {nombre}: disparó, pero no por eso\n{salida}")
            else:
                print(f"ok  {nombre}")

    if fallos:
        print(f"\n{len(fallos)} regla(s) del guardia no sirven todavía")
        return 1
    print(f"\nlas {len(CASOS)} reglas del guardia disparan cuando tienen que disparar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
