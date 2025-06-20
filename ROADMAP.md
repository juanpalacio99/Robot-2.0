# Roadmap del Proyecto Robot 2.0

Este roadmap consolida los 8 bloques de trabajo de la Fase 1.0 y sirve como guía para el desarrollo e integración en el repositorio de GitHub.

---

## 1. Plataforma de Datos Empresarial + Validación & Caché

* **Lakehouse con gobernanza** (Delta Lake o Apache Iceberg)
* **Ingesta de datos en tiempo real**: Kafka o AWS Kinesis
* **Feature Store híbrido**:

  * *Batch*: PySpark
  * *Online*: Feast + Redis (caché low‑latency)
* **Validación de datos**: Great Expectations / Evidently
* **Detección de drift**: Alibi Detect

**Entregables**:

* Esquema de la lakehouse y scripts de ingestión
* Pipelines de validación automatizados
* Servicio de cacheo en Redis

---

## 2. MLOps & Tracking

* **Orquestación**: Argo Workflows o Prefect
* **CI/CD de modelos** (validación + retraining automático)
* **Registro y seguimiento**: MLflow + Weights & Biases
* **Monitorizado de drift y calidad**
* **Shadow deployments & pruebas A/B**

**Entregables**:

* Definición de pipelines en Argo/Prefect
* Integración con MLflow + W\&B
* Reportes de monitorización en Grafana

---

## 3. Agente Core + Offline RL

* Arquitectura: Double‑Dueling DQN + Prioritized Replay
* Entrenamiento offline con datos históricos de mercado
* Aceleración con `tf.function`
* Simulador de eventos críticos (2008, 2020…)
* Early‑stopping y métricas: Sharpe, Sortino

**Entregables**:

* Código del agente DQN entrenable
* Simulador integrado en pipeline
* Notebook de evaluación de métricas

---

## 4. Safe RL + XAI + Stress Testing

* **Constrained PPO** (penalizaciones drawdown)
* **Adversarial training** (latencia, slippage)
* **Explainability**: SHAP, mapas de atención
* **Clusterización de regímenes de mercado**

**Entregables**:

* Clases de entrenamiento seguro
* Reportes XAI automáticos
* Casos de stress-testing

---

## 5. Auto‑ML, Meta‑RL & Auto‑NAS

* Tuning distribuido: Optuna + Ray Tune
* Meta‑RL (MAML, RL²)
* Auto‑NAS guiado por recompensa

**Entregables**:

* Scripts de búsqueda automatizada
* Estudios comparativos de arquitecturas

---

## 6. Modelos Avanzados

* GNN para relaciones cross‑activo
* Transformers temporales
* MARL para portafolio y cobertura
* Factores macroeconómicos integrados

**Entregables**:

* Implementaciones de GNN y Transformers
* Evaluación de MARL en portafolios

---

## 7. Despliegue & App Móvil

* **Microservicios**: Docker + Kubernetes (K3s/minikube)
* **Backend**: FastAPI + gRPC + OAuth
* **Monitoring**: Prometheus + Grafana
* **App React Native**: historial, simulador, alertas
* Failover + rollback

**Entregables**:

* Manifiestos Kubernetes
* FastAPI con auth y gRPC
* App móvil básica

---

## 8. Bonus: Estrategia de Mercado Institucional

* Detección de eventos macro con NLP
* Score de calidad de mercado (volatilidad, spread)
* Evaluación ética y anti‑manipulación
* Autoevaluación semanal con IA generativa

**Entregables**:

* Pipelines NLP para eventos macro
* Dashboard de calidad de mercado

---

