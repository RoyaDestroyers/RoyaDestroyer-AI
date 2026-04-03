# RoyaDestroyer-AI V1

Estado oficial del modulo IA al cierre de la v1.

## Modelo canonico

- version: `roya-model-v1.0.0`
- framework: `PyTorch`
- arquitectura: `mobilenetv3_large_100`
- tamano de entrada: `224x224`
- inferencia: solo en servidor

Artefacto oficial:
- `artifacts/models/roya-model-v1.0.0/model.pt`

Metadata oficial:
- `artifacts/models/roya-model-v1.0.0/metadata.json`
- `artifacts/models/roya-model-v1.0.0/labels.json`
- `artifacts/models/roya-model-v1.0.0/metrics.json`

## Clases v1

- `cercospora`
- `hoja_sana`
- `minador`
- `phoma`
- `roya_avanzada`
- `roya_temprana`

## Dataset usado en v1

Fuentes integradas en la v1:
- `CLR EAFIT`
- `JMuBEN`
- `JMuBEN2`
- `Rust+Miner Brazil`

Fuentes pendientes para versiones posteriores:
- `Uganda`
- `RoCoLe`
- `BRACOL`

Distribucion limpia usada para entrenar la v1:
- `cercospora`: `322`
- `hoja_sana`: `3228`
- `minador`: `1896`
- `phoma`: `691`
- `roya_avanzada`: `1131`
- `roya_temprana`: `1590`

## Resultado de la v1

Metricas en test:
- `accuracy`: `0.9917541229385307`
- `macro_f1`: `0.9906208727134705`
- `weighted_f1`: `0.9917576258047558`
- `recall_roya_temprana`: `0.9874476987447699`
- `recall_roya_avanzada`: `0.9766081871345029`

Archivos de soporte:
- `artifacts/metrics/train_history.json`
- `artifacts/metrics/evaluation.json`

## Servicio IA

Entrypoint:
- `service/app/main.py`

Endpoints:
- `GET /health`
- `POST /predict`

El servicio actual carga correctamente el modelo oficial de 6 clases.

## Estado del repo

Este repo queda listo para:
- seguir mejorando datasets en una v1.1 o v2
- integrarse con el backend principal
- dockerizar despliegue del servicio IA

No hace falta volver a definir la v1 del modelo. La referencia oficial es este archivo.
