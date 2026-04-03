# RoyaDestroyer-AI

Modulo de IA para RoyaDestroyer. Este repo interno contiene:

- preparacion y limpieza de datasets
- entrenamiento y evaluacion del modelo
- exportacion de artefactos versionados
- servicio de inferencia para ejecutar en servidor
- documentacion tecnica de integracion con el backend principal

## Alcance v1

Clasificacion multiclase de imagenes de hojas de cafe con estas etiquetas:

- `roya_temprana`
- `roya_avanzada`
- `cercospora`
- `phoma`
- `minador`
- `hoja_sana`
- `imagen_invalida`

La inferencia corre solo en servidor. El frontend nunca ejecuta el modelo.

## Estructura

```text
RoyaDestroyer-AI/
  data/          datasets y reportes
  scripts/       pipeline ejecutable
  src/           libreria de entrenamiento/inferencia
  service/       FastAPI para despliegue
  artifacts/     modelos y metricas exportadas
  docs/          especificacion del modulo
```

## Inicio rapido

1. Crear entorno Python 3.10
2. Instalar dependencias desde `requirements.txt`
3. Instalar el paquete en modo editable: `pip install -e .`
4. Leer `scripts/download_instructions.md`
5. Colocar datasets en `data/raw/`
6. Ejecutar:

```bash
python scripts/verify_environment.py
python scripts/unify_datasets.py
python scripts/verify_images.py
python scripts/dedup.py
python scripts/report_distribution.py
python scripts/split.py
python scripts/train.py
python scripts/evaluate.py
python scripts/export_model.py
```

## Notas
- El entrenamiento esta pensado para laptop local primero. Si la GPU no esta disponible, el baseline puede correr en CPU con lotes mas pequenos.
