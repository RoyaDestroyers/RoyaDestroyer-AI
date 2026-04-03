# API Contract

## Health

`GET /health`

Response:

```json
{
  "status": "ok",
  "modelLoaded": true,
  "modelVersion": "roya-model-v1.0.0",
  "classes": 7
}
```

## Predict

`POST /predict`

Request with multipart form:

- `image`: binary file
- `lote_id`: optional string
- `observaciones`: optional string

Response:

```json
{
  "predictedClass": "roya_temprana",
  "confidence": 0.91,
  "topK": [
    {"label": "roya_temprana", "score": 0.91},
    {"label": "roya_avanzada", "score": 0.06},
    {"label": "hoja_sana", "score": 0.03}
  ],
  "severity": "moderado",
  "symptoms": ["manchas amarillas", "polvillo naranja en enves"],
  "recommendations": ["aislar hojas afectadas", "aplicar manejo fitosanitario"],
  "modelVersion": "roya-model-v1.0.0",
  "isInvalidImage": false
}
```
