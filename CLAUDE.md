# ATLAS: instrucciones para cualquier sesión que trabaje en este repo

Este fichero lo lee automáticamente cualquier sesión de Claude Code abierta sobre
este repositorio, en local o en la nube. **Es la fuente de la verdad del proyecto.**
No dependas de la memoria de ninguna cuenta: si algo importa, vive aquí o en `docs/`.

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

Lo mismo con `docs/estado.md` cuando publiques un artículo: tabla, color de acento
y lo que quede pendiente.

## Documentación

- **[docs/receta-de-articulo.md](docs/receta-de-articulo.md)**: qué es fijo y qué
  cambia en cada artículo, y la lista larga de fallos ya pagados. Léelo entero
  antes de escribir un artículo nuevo.
- **[docs/verificacion.md](docs/verificacion.md)**: cómo se verifica de verdad en
  el navegador, y las trampas del entorno que producen falsos "todo correcto".
- **[docs/estado.md](docs/estado.md)**: qué artículos hay, con qué color, qué se
  midió en cada uno y qué viene después.
- **[docs/despliegue.md](docs/despliegue.md)**: commit, push y comprobación.

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
