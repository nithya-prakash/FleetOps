#!/bin/sh
# Start MLflow with basic-auth. Credentials come from secret files and are exported only into this process.
set -eu
umask 077
# The admin password goes into a private (0600, tmpfs) copy of the config rather than the process environment.
{ cat /app/basic_auth.ini; printf 'admin_password = %s\n' "$(cat /run/secrets/mlflow_admin_password)"; } > /tmp/basic_auth.ini
export MLFLOW_AUTH_CONFIG_PATH=/tmp/basic_auth.ini
# MLflow offers no file-based option for the Flask session key; it must be an environment variable of this process.
MLFLOW_FLASK_SERVER_SECRET_KEY="$(cat /run/secrets/mlflow_flask_secret)"
export MLFLOW_FLASK_SERVER_SECRET_KEY
# HTTPS when a certificate is mounted (compose.tls.yaml): uvicorn serves the certificate, clients verify it.
TLS_OPTS=""
if [ -f /run/secrets/tls/server.crt ]; then
  TLS_OPTS="--ssl-certfile /run/secrets/tls/server.crt --ssl-keyfile /run/secrets/tls/server.key"
fi
# shellcheck disable=SC2086
exec mlflow server --host 0.0.0.0 --port 5000 --app-name basic-auth ${TLS_OPTS:+--uvicorn-opts "$TLS_OPTS"} \
  --backend-store-uri sqlite:////data/mlflow.db \
  --serve-artifacts --artifacts-destination /data/artifacts --default-artifact-root mlflow-artifacts:/ \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-mlflow:5000,localhost:5000,127.0.0.1:5000,localhost:15000,127.0.0.1:15000}"
