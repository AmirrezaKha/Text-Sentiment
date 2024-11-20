FROM debian:buster-slim

RUN apt-get update && apt-get install -y wget
RUN wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio_20240510014138.0.0_amd64.deb -O minio.deb && \
  dpkg -i minio.deb && \
  mkdir ~/minio

EXPOSE 9000 9090
