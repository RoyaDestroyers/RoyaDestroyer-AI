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
