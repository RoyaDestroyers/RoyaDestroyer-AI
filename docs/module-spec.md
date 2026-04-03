# Module Spec

## Goal

Resolver el flujo completo de IA para RoyaDestroyer:

1. recibir imagen
2. clasificar enfermedad/condicion
3. enriquecer con severidad, sintomas y recomendaciones
4. entregar una respuesta estable al backend principal

## Scope v1

- clasificacion multiclase
- inferencia solo en servidor
- modelo base con transfer learning en `PyTorch`
- servicio HTTP interno

## Out of Scope v1

- inferencia en cliente
- entrenamiento distribuido
- segmentacion
- deteccion de objetos como producto final
- clase `ojo_de_gallo`
