# Publicar y comprobar el despliegue

> Documento de trabajo del repositorio. Las reglas duras y el índice están
> en [CLAUDE.md](../CLAUDE.md). **Si descubres un fallo nuevo, añádelo aquí
> en la misma sesión.**


En el repo ATLAS: tras commitear, **hacer push a origin/main siempre, sin pedir confirmación**, y después **verificar la web real desplegada** en https://aleetreny.github.io/ATLAS/ (GitHub Pages ya activo, main → `/(root)`, que es donde vive el sitio; `docs/` es documentación de trabajo y no se publica).

**Why:** El usuario lo pidió explícitamente ("haz el push siempre, sin preguntarme y revisando la web real"). Pages tarda ~1-2 min en desplegar tras el push.

**How to apply:** commit → push → esperar/poll el deploy → navegar la URL pública y comprobar consola/DOM. Además: firmas de artículos = "aleetreny" sin fecha; evitar logos/emblemas decorativos (el compass rose le pareció "muy IA" y se eliminó — favicon es una "A" tipográfica simple). Relacionado: [el estado del proyecto](estado.md).
