# Download Instructions

La ruta recomendada ya no es manual. El repo incluye `scripts/download_datasets.py`
para descargar y extraer los datasets publicos compatibles.

## Destinos esperados

- `data/raw/jmuben/`
- `data/raw/jmuben2/`
- `data/raw/clr_eafit/`
- `data/raw/bracol/`
- `data/raw/rocole/`
- `data/raw/uganda/`
- `data/raw/rust_miner_brazil/`
- `data/raw/roboflow_optional/`
- `data/raw/own_invalid/`

## Orden recomendado

1. JMuBEN / JMuBEN2
2. CLR EAFIT
3. BRACOL
4. RoCoLe
5. Uganda
6. Invalid images propias

## Uso

```bash
python scripts/download_datasets.py jmuben jmuben2 bracol rocole uganda rust_miner_brazil
python scripts/download_datasets.py clr_eafit
```

Si quieres bajar todo de una vez:

```bash
python scripts/download_datasets.py all
```

Opciones utiles:

```bash
python scripts/download_datasets.py all --no-extract
```

Los zips de Mendeley quedan en `data/raw/_archives/` y cada dataset se extrae
en su carpeta correspondiente dentro de `data/raw/`.

## Estado actual

- `jmuben`: descargado y extraido
- `jmuben2`: descargado y extraido
- `clr_eafit`: clonado desde GitHub
- `uganda`: descarga automatizada disponible, pero tarda por venir como miles de JPG pequenos
- `rust_miner_brazil`: descarga automatizada disponible
- `rocole`: descarga automatizada disponible; requiere tiempo por volumen de imagenes
- `bracol`: el archivo publicado por Mendeley llega corrupto y queda marcado como error en `data/reports/download_manifest.json`

## Regla

No mezclar archivos entre datasets. Cada dataset debe quedar descomprimido dentro de su carpeta correspondiente.
