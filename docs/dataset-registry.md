# Dataset Registry

## Base Datasets

- JMuBEN
- JMuBEN2
- CLR EAFIT
- BRACOL
- RoCoLe
- Uganda Coffee Leaf Disease Dataset

## Optional Datasets

- Rust+Miner Brazil
- Roboflow derivatives

## Class Mapping v1

- Rust -> `roya_temprana` unless a source already provides severity levels
- Rust levels 3-4 -> `roya_avanzada`
- Healthy -> `hoja_sana`
- Invalid captures -> `imagen_invalida`

## Acquisition Status

- `jmuben`: listo localmente
- `jmuben2`: listo localmente
- `clr_eafit`: listo localmente
- `uganda`: pipeline de descarga automatizado; adquisicion en progreso por volumen de archivos pequenos
- `rust_miner_brazil`: pipeline de descarga automatizado
- `rocole`: pipeline de descarga automatizado
- `bracol`: bloqueado por archivo fuente invalido en Mendeley
