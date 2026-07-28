# Publicar y comprobar el despliegue

> Documento de trabajo del repositorio. Las reglas duras y el índice están
> en [CLAUDE.md](../CLAUDE.md). **Si descubres un fallo nuevo, añádelo aquí
> en la misma sesión.**


En el repo ATLAS: tras commitear, **hacer push a origin/main siempre, sin pedir confirmación**, y después **verificar la web real desplegada** en https://aleetreny.github.io/ATLAS/ (GitHub Pages ya activo, main → `/(root)`, que es donde vive el sitio; `docs/` es documentación de trabajo y no se publica).

**Why:** El usuario lo pidió explícitamente ("haz el push siempre, sin preguntarme y revisando la web real"). Pages tarda ~1-2 min en desplegar tras el push.

**How to apply:** commit → push → esperar/poll el deploy → navegar la URL pública y comprobar consola/DOM. Además: firmas de artículos = "aleetreny" sin fecha; evitar logos/emblemas decorativos (el compass rose le pareció "muy IA" y se eliminó: el favicon es una "A" tipográfica simple). Relacionado: [el estado del proyecto](estado.md).

## Varias sesiones a la vez: ramas, números y colores

Una sesión en la nube trabaja en su propia rama (`claude/<tema>-<id>`) y **Pages
sirve `main`**, así que hasta que la rama no entre en main no hay nada
desplegado. Con más de una sesión abierta eso deja de ser un detalle:

El 2026-07-28 había tres ramas vivas a la vez y **las tres numeraron su artículo
como el 27**, dos eligieron el mismo acento (`#7e22ce`), y las tres editaron
`docs/estado.md`, `index.html` y `docs/receta-de-articulo.md`. Git no avisa de
nada de eso: los ficheros compartidos dan conflicto de texto normal, se
resuelven a mano, y una resolución a mano puede dejar dos artículos con el mismo
número o un chip apuntando a una carpeta que no existe. **El trabajo no se
pierde nunca** (las ramas siguen en origin y un conflicto no borra nada), pero
la portada puede quedar mintiendo, y eso sí se publica.

**Quien lanza las sesiones reparte antes de empezar**, con una sola orden, y
pega en el prompt de cada una la línea que le toca:

```
python src/utils/check_publication.py --reparte 3
```

Eso da a cada sesión su número y su acento ya asignados. Es lo único que cierra
la carrera de verdad: `--next` no vale con varias a la vez, porque tres sesiones
que arrancan el mismo día leen las tres el mismo "siguiente libre" y vuelven a
chocar, solo que ahora el guardia las caza en el push, con el artículo escrito.
Un reparto hecho antes de empezar no tiene carrera que perder.

El procedimiento, para cada rama que entre después de la primera:

1. `git fetch origin main && git merge origin/main` **sobre la rama**, no al
   revés: los conflictos se resuelven en la rama, donde está el contexto.
2. `python src/utils/check_publication.py --next` dice **qué número toca y qué
   acentos quedan libres**. Renumerar la fila de `estado.md`, repintar
   `--primary`/`--secondary` si el color estaba cogido, y regenerar la miniatura
   si dependía del acento.
3. Releer el párrafo de "qué viene después" de `estado.md`: lo escribió quien
   entró antes y casi siempre queda obsoleto al entrar el siguiente.
4. `python src/utils/check_publication.py` en verde antes de pushear.
5. Merge a main y comprobar la web.

**El guardia**: `src/utils/check_publication.py` comprueba las nueve cosas que
esta clase de choque rompe (números repetidos o con huecos, filas que citan
carpetas inexistentes, artículos sin fila, acentos repetidos, la tabla
discrepando del `--primary` real, tarjetas o miniaturas que faltan, chips vivos a
la nada, y rayas largas). `src/utils/test_check_publication.py` monta un sitio de
mentira y rompe una cosa cada vez para exigir que **cada regla dispare**, porque
un guardia que calla solo vale si se ha comprobado que sabe hablar. Los dos
corren en `.github/workflows/publicacion.yml` en cada push y en cada pull
request contra main, en ese orden.

Lo que el guardia NO puede hacer es evitar que dos sesiones elijan el mismo
número mientras escriben a la vez, porque no se ven entre ellas. Lo que hace es
que ese choque sea **imposible de publicar sin enterarse**: falla en rojo en el
push de la rama, mucho antes del merge. La costumbre que lo evita del todo es
correr `--next` al empezar un artículo, no al terminarlo.
