# ATLAS: cómo se trabaja en este repositorio

Este fichero es lo primero que se lee al abrir el repo, en local o en la nube.
**Es la fuente de la verdad del proyecto.** No dependas de notas guardadas fuera
del repositorio: si algo importa, vive aquí o en el resto de `docs/`.

## Qué es esto

Una web educativa interactiva publicada en <https://aleetreny.github.io/ATLAS/>,
con la estética de mlu-explain.github.io. Sitio estático en la **raíz del repo**
(`index.html` + `assets/` + una carpeta por artículo), sin build step: HTML a mano,
ES modules, D3 v7 y KaTeX vendorizados. GitHub Pages sirve main desde `/(root)`,
con `.nojekyll`.

Los artículos se escriben **en inglés**. Esta documentación de trabajo está en
español, que es el idioma en el que se lleva el proyecto.

## Las tres reglas duras

1. **Toda cifra se mide, nunca se cita.** Cada número que aparece en una página
   sale de un generador en `src/utils/generate_*.py` que lo calcula de verdad. Si
   falta una librería, se instala. Si un resultado contradice el guion previsto,
   **se reescribe el guion, no el resultado**: ahí suele estar el mejor artículo.
2. **Ni rayas largas ni medias** (— ni –) en nada: prosa, comentarios, mensajes de
   commit, documentación. Se usan comas, dos puntos, paréntesis, o se parte la frase.
   Los rangos numéricos con guion normal.
3. **Los commits son de `aleetreny <alejandrotreny100@gmail.com>` y de nadie más.**
   Sin trailers de coautoría, sin líneas de "generado con", sin `--author` apuntando
   a otro sitio. Y se pushea siempre, sin preguntar, verificando después el
   despliegue real.

## La regla de mantenimiento (importante)

**Cada vez que descubras un fallo nuevo, anótalo en `docs/receta-de-articulo.md`
(si es de contenido, matemática, layout o interacción) o en `docs/verificacion.md`
(si es del entorno o del proceso de comprobación), en la misma sesión en que lo
descubres.** Con una línea que diga qué pasó, cómo se detectó y cuál es la regla
para no repetirlo. Ese fichero es la razón de que cada artículo salga mejor que el
anterior, y solo funciona si se actualiza en caliente.

**Y no basta con anotarlo: se arregla.** Ninguna edición se da por terminada
mientras quede un fallo que hayas visto al interactuar con la web, sea estético
(algo que se ve mal), lógico (algo que dice o hace lo que no es) o didáctico (algo
que enseña mal o confunde), esté o no dentro del encargo original y esté o no en el
artículo que estabas tocando. Si el arreglo es grande o cambia páginas ya
publicadas, dilo antes de hacerlo, pero hazlo.

Lo mismo con `docs/estado.md` cuando publiques un artículo: tabla, color de acento
y lo que quede pendiente.

## El número, el acento, y el merge a main

Las dos cosas se reparten distinto, y el motivo es cuándo se fijan:

- **El acento te lo da el encargo.** Va dentro del artículo desde la primera
  línea que escribes, así que no puede esperar. Si el encargo trae un
  `--primary`, es ese: quiere decir que hay otras sesiones escribiendo a la vez
  y que el reparto ya está hecho. No lo elijas ni lo deduzcas del repositorio.
- **El número lo coges al mergear**, no al empezar. Vive solo en la tabla de
  `docs/estado.md` y ninguna página lo enseña, así que fijarlo antes no aporta
  nada y sí crea una carrera: tres sesiones que arrancan el mismo día leen las
  tres el mismo "siguiente libre". Al mergear no hay carrera, porque main
  serializa.

```
python src/utils/check_publication.py --next   # el número libre AHORA, y qué acentos quedan
python src/utils/check_publication.py          # ¿es coherente lo que voy a publicar?
```

Ya pasó lo contrario: tres ramas reclamando el 27 el mismo día y dos el acento
`#7e22ce`. Quien lanza las sesiones reparte los acentos con `--reparte N` antes
de empezar, y ese reparto **mira también las ramas sin mergear**, no solo lo
publicado. El guardia es la red de abajo, no el reparto.

**Autorización permanente del dueño del repo para cerrar el trabajo (no hace
falta preguntar cada vez):** cuando el artículo esté verificado, **mergea tu
propia rama a `main` y pushea**, que es lo que lo publica, porque Pages sirve
`main` desde `/(root)`. El procedimiento exacto, con lo que hay que hacer cuando
main se ha movido mientras escribías, está en
[docs/despliegue.md](despliegue.md). En corto:

```
git fetch origin main && git merge origin/main   # sobre TU rama, primero
python src/utils/test_check_publication.py       # el guardia sabe disparar
python src/utils/check_publication.py            # y calla sobre tu árbol
git checkout main && git merge --ff-only <tu-rama> && git push origin main
```

Las dos comprobaciones van **antes** del merge y las dos tienen que salir en
verde: son la única condición. Si alguna falla, se arregla, no se mergea. Y si
el merge de `origin/main` sobre tu rama da conflicto en `docs/estado.md`,
`index.html` o `docs/receta-de-articulo.md`, es lo normal con varias sesiones:
se resuelve quedándote con lo de main y volviendo a aplicar lo tuyo encima,
nunca al revés.

## Documentación

- **[docs/receta-de-articulo.md](receta-de-articulo.md)**: qué es fijo y qué
  cambia en cada artículo, y la lista larga de fallos ya pagados. Léelo entero
  antes de escribir un artículo nuevo.
- **[docs/verificacion.md](verificacion.md)**: cómo se verifica de verdad en
  el navegador, y las trampas del entorno que producen falsos "todo correcto".
- **[docs/estado.md](estado.md)**: qué artículos hay, con qué color, qué se
  midió en cada uno y qué viene después.
- **[docs/despliegue.md](despliegue.md)**: commit, push y comprobación.

## Cómo se construye un artículo, en corto

1. **Generador** en `src/utils/generate_<tema>_data.py`. Sembrado, con
   comparaciones controladas (cambiar una cosa, fijar el resto), que cachee cada
   experimento terminado fuera del repo (`~/.atlas_vision_data/<algo>_cache/`) para
   que una interrupción no cueste la tirada entera, y que exporte a
   `<articulo>/data/*.json` **todas** las cifras que la página vaya a citar.
2. **Widgets** en `<articulo>/js/`, arrancados uno a uno dentro de un `safely()`
   en `main.js` para que un fallo no tumbe al resto. Helpers compartidos en
   `assets/js/chart.js` (SVG), `assets/js/imagery.js` (canvas y píxeles) y
   `assets/js/scrolly.js`.
3. **Prosa numérica compuesta desde el JSON en tiempo de carga** (`<articulo>/js/prose.js`),
   no escrita a mano, y con ramas condicionales en toda comparación: si se
   re-ejecuta el generador y un resultado se mueve, el texto se mueve con él en vez
   de quedarse mintiendo. Este patrón nació del artículo 20, donde una tabla se
   quedó con un tiempo obsoleto.
4. **Verificación**: script que cruza las cifras del HTML contra el JSON, más
   barrido en navegador (cada botón, cada slider de extremo a extremo, cada paso de
   scrolly, a 1400px y a 375px) desde un **puerto frío**, porque el navegador cachea
   los módulos ES y te enseña la versión vieja.
5. **Publicar**: tarjeta en la portada dentro de la galería de su módulo, chip vivo
   en el mapa de ramas, cierre del artículo anterior apuntando al nuevo, commit,
   push, y comprobar la web desplegada.

## Datos pesados

Los datasets viven **fuera del repo**, en `~/.atlas_vision_data`, y los descarga
`src/utils/vision_data.py` la primera vez (`mnist()`, `fashion_mnist()`,
`cifar10()`). Los cachés de entrenamiento también. Nada de eso se commitea: un
clon limpio reproduce cualquier cifra ejecutando su generador.

Con un límite que conviene saber: **la reproducción es exacta en la misma
máquina, no entre máquinas.** El número de hilos cambia el orden de las
reducciones en coma flotante, así que una precisión se mueve en el quinto
decimal, y un tiempo de reloj se mueve por completo (4,4 s aquí, 42,8 s en un
contenedor de 4 núcleos). Por eso los generadores se ejecutan siempre en la
misma máquina y un tiempo se publica como razón entre brazos, nunca como
constante. Ver la trampa 14 de [docs/verificacion.md](verificacion.md).
