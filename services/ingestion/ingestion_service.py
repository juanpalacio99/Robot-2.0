"""
ingestion_service.py

Propósito:
    Servicio de ingesta de datos en tiempo real desde Kafka hacia el sistema.

Dependencias:
    - kafka-python>=2.0

Ejemplo de uso:
    1. Arrancar Kafka y Zookeeper mediante Docker Compose (ver descripción abajo).
    2. Ejecutar el servicio:
       python /services/ingestion/ingestion_service.py
    3. Enviar mensajes de prueba al tópico `raw_data`:
       kafka-console-producer --broker-list localhost:9092 --topic raw_data
       > {"symbol": "EURUSD", "price": 1.2345, "time": 1620000000}

Según Sección 4 (Módulo 1) de **Instrucciones.pdf**, este módulo debe implementar “Ingesta de datos en tiempo real” mediante Kafka o Kinesis, y exponer pruebas unitarias que simulen la ingesta. Además, según Sección 1 de **Fase 1.0.pdf**, se requiere la capa de “Ingesta de datos en tiempo real: Kafka o AWS Kinesis” .
"""

# Bloque de comandos de prueba (Docker/pytest) comentado:
# 1. Levantar Zookeeper y Kafka con Docker Compose:
#    docker-compose -f docker-compose.kafka.yml up -d
#
# 2. Verificar que Kafka esté accesible en localhost:9092.
#
# 3. Ejecutar el servicio de ingesta:
#    python /services/ingestion/ingestion_service.py
#
# 4. Enviar un mensaje de prueba:
#    kafka-console-producer --broker-list localhost:9092 --topic raw_data
#    > {"symbol":"EURUSD","price":1.2345,"time":1620000000}
#
# 5. Observar en consola que el mensaje haya sido recibido.

from kafka import KafkaConsumer
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    """
    Conecta al broker de Kafka y consume mensajes del tópico 'raw_data'.
    Actualmente, simplemente imprime cada mensaje recibido.
    En un entorno productivo, aquí se podría insertar cada registro en un
    sistema de almacenamiento o feature store downstream.
    """
    consumer = KafkaConsumer(
        'raw_data',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='ingestion_group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    logging.info("Ingestion service iniciado. Escuchando el tópico 'raw_data'...")
    for message in consumer:
        data = message.value
        # Procesamiento mínimo: imprimir en stdout
        logging.info(f"Mensaje recibido: {data}")
        # TODO: reemplazar con lógica para almacenar/escribir datos en almacenamiento intermedio

if __name__ == "__main__":
    main()
