"""
test_ingestion.py

Pruebas unitarias para el módulo de ingesta de datos en tiempo real.
Simula una producción de mensaje en un tópico Kafka 'test_raw_data' y
verifica que un consumidor simple lo reciba correctamente.

Dependencias de pruebas:
    - pytest
    - kafka-python>=2.0

Ejemplo de uso:
    1. Levantar Kafka y Zookeeper:
       docker-compose -f docker-compose.kafka.yml up -d
    2. Ejecutar:
       pytest /tests/test_ingestion.py
"""

import threading
import time
import pytest
import json
from kafka import KafkaProducer, KafkaConsumer

@pytest.fixture(scope="module")
def kafka_producer():
    """
    Crea un productor Kafka apuntando a localhost:9092.
    """
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    yield producer
    producer.close()

def run_consumer(stop_event):
    """
    Función para ejecutar un consumidor que escuche 'test_raw_data'
    y detenga el hilo al recibir el primer mensaje.
    """
    consumer = KafkaConsumer(
        'test_raw_data',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='test_ingestion_group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for message in consumer:
        data = message.value
        print(f"Test ingestion recibió: {data}")
        stop_event.set()
        break

def test_ingestion_flow(kafka_producer, capsys):
    """
    1. Inicia un hilo con un consumidor que escuche 'test_raw_data'.
    2. Envía un mensaje de prueba al tópico.
    3. Espera a que el consumidor imprima el mensaje y verifica la salida.
    """
    stop_event = threading.Event()
    consumer_thread = threading.Thread(target=run_consumer, args=(stop_event,))
    consumer_thread.start()

    # Esperar a que el consumidor se inicialice
    time.sleep(1)

    # Enviar mensaje de prueba
    test_message = {"symbol": "EURUSD", "price": 1.2345, "time": 1620000000}
    kafka_producer.send('test_raw_data', test_message)
    kafka_producer.flush()

    # Esperar hasta 10s a que el consumidor procese el mensaje
    flag = stop_event.wait(timeout=10)
    captured = capsys.readouterr()

    assert flag is True, "El consumidor no recibió el mensaje en el tiempo esperado."
    assert "Test ingestion recibió" in captured.out, "No se encontró la impresión del mensaje recibido."

    consumer_thread.join(timeout=1)
