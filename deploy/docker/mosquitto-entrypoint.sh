#!/bin/sh
# Copy secrets into a private tmpfs (owner-only) because Mosquitto insists on strict file permissions, then start.
#   /run/secrets/mosquitto_passwd           password file (always)
#   /run/secrets/tls/{ca.crt,server.crt,server.key}   when a TLS profile is selected via MOSQUITTO_CONF
set -eu
umask 077
cp /run/secrets/mosquitto_passwd /mosquitto/run/passwd
chmod 600 /mosquitto/run/passwd
if [ -d /run/secrets/tls ]; then
  mkdir -p /mosquitto/run/tls
  for f in ca.crt server.crt server.key; do
    cp "/run/secrets/tls/$f" "/mosquitto/run/tls/$f"
    chmod 600 "/mosquitto/run/tls/$f"
  done
fi
exec /usr/sbin/mosquitto -c "${MOSQUITTO_CONF:-/mosquitto/config/mosquitto.conf}"
