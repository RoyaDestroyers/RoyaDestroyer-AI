# CI/CD

## CI

El pipeline de integracion ejecuta:
- instalacion del paquete con `pip install -e .`
- verificacion del entorno con `scripts/verify_environment.py`
- compilacion de fuentes con `compileall`
- pruebas unitarias con `unittest`

Workflow:
- `.github/workflows/ci.yml`

## Deploy

El pipeline de despliegue usa SSH para entrar a la VPS y ejecutar `docker compose up -d --build` en el directorio de despliegue.

Workflow:
- `.github/workflows/deploy.yml`

### Secretos requeridos
- `SSH_HOST`
- `SSH_USER`
- `SSH_PRIVATE_KEY`
- `DEPLOY_PATH`

### Convenciones
- `DEPLOY_PATH` debe apuntar al directorio donde el repo ya esta clonado en la VPS.
- el deploy asume `main` como rama de referencia.
- los secretos no deben versionarse ni hardcodearse en el workflow.

## Nota operativa
El servicio IA esta pensado para correr detras de `docker compose`. El workflow de despliegue solo actualiza el codigo y levanta los contenedores; la configuracion de red, volumenes y variables de entorno se hace fuera del repositorio mediante secretos y archivos `.env` del servidor.
