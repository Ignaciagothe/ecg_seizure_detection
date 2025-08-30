# Detección de Crisis Epilépticas en ECG

Implementaciones en **PyTorch** para detectar eventos epilépticos a partir de registros de ECG. Incluye scripts de preprocesamiento, entrenamiento y evaluación con dos enfoques de modelo:

| Enfoque | Descripción | Cuándo usar |
|---------|-------------|-------------|
| **Modelo jerárquico** | Extrae ventanas fijas, genera embeddings con un encoder CNN (p. ej. InceptionTime) y modela la secuencia de ventanas con un GRU/LSTM/Transformer. | Cuando ya tienes las ventanas preprocesadas y quieres experimentar con distintos modelos secuenciales. |
| **Pipeline híbrido Transformer-CNN** | Segmenta automáticamente los archivos brutos. Un CNN aprende características locales; un Transformer (con self-attention y codificación posicional) captura dependencias de largo plazo y toma la decisión final. | Para un flujo *end-to-end* desde los CSV originales o cuando quieras evitar pasos de preprocesamiento manual. |

---

## Tabla de contenidos
1. [Descripción general](#descripción-general)  
2. [Estructura del repositorio](#estructura-del-repositorio)  
3. [Requisitos](#requisitos)  
4. [Instalación](#instalación)  
5. [Preparación de datos](#preparación-de-datos)  
6. [Entrenamiento](#entrenamiento)  
   - [Modelo jerárquico](#modelo-jerárquico)  
   - [Pipeline híbrido Transformer-CNN](#pipeline-híbrido-transformer-cnn)  


## Descripción 

Este proyecto muestra cómo detectar crisis epilépticas a partir de señales ECG de un solo electrodo. Se proporcionan:

- Scripts de **preprocesamiento** para convertir trazas crudas (`*.csv`) en ventanas constantes (`*.npz`).
- **Modelos** que combinan CNNs, RNNs y Transformers para capturar tanto la morfología local como la dinámica temporal de largo plazo.
- Script (utils) con funciones para entrenar y validar los modelos, así como para registrar las métricas y pesos durante el entrenamiento

> **Nota de datos**: Los archivos ECG brutos **no** se incluyen.  
> Cada CSV debe tener al menos las columnas `Time [s]`, `Signal [mV]`, `Seizure [bool]`.

---

## Estructura del repositorio

```
.
├── src/                     # Datasets, preprocesamiento y arquitecturas
├── train\_hierarchical.py    # Entrenamiento con ventanas preprocesadas
├── train\_hybrid\_pipeline.py # Pipeline end-to-end desde archivos brutos
├── notebooks/               # Exploración y visualización
└── requirements.txt         # Dependencias de Python

````



## Instalación

```bash
git clone https://github.com/Ignaciagothe/ecg_seizure_detection.git
cd ecg_seizure_detection
pip install -r requirements.txt
````


## Preparación de datos

Convierte las trazas crudas en ventanas de 5 s (ajusta si es necesario):

```bash
python -m src.preprocessing \
    --data_dir data/raw_ecg \
    --file_list data/splits/train.txt \
    --out_path data/processed/windows_train.npz \
    --window_seconds 5
```

Repite para **train**, **val** y **test**.


## Entrenamiento

### Modelo jerárquico

```bash
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz   data/processed/windows_val.npz \
    --test_npz  data/processed/windows_test.npz \
    --out_dir   runs/hierarchical
```

* Métricas → `runs/hierarchical/metrics.csv`
* Mejores pesos → `runs/hierarchical/best_model.pt`

### Pipeline híbrido Transformer-CNN

```bash
python train_hybrid_pipeline.py \
    --data_dir data/raw_ecg \
    --out_dir  runs/hybrid
```

Si no existen, el script creará automáticamente los splits de **train**, **val** y **test**.

---

## 📚 Documentación Completa

Para una explicación detallada de la arquitectura, algoritmos y modelos:

- **[ARCHITECTURE_DOCUMENTATION.md](ARCHITECTURE_DOCUMENTATION.md)** - Documentación técnica completa
- **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Resumen ejecutivo de los algoritmos
- **[test_architectures.py](test_architectures.py)** - Script de prueba y visualización de modelos

### Prueba rápida de arquitecturas:
```bash
python test_architectures.py
```

---

## 🏗️ Arquitecturas Implementadas

1. **InceptionTime + Modelo Jerárquico**: CNN multi-escala con modelado secuencial
2. **Pipeline Híbrido Transformer-CNN**: Procesamiento end-to-end con atención
3. **Temporal Convolutional Network (TCN)**: Convoluciones dilatadas causales  
4. **Transformer Puro**: Modelado de secuencias con auto-atención

### Métricas de evaluación:
- **AUROC/AUPRC**: Calidad de ranking y precisión en datos desbalanceados
- **F1/Precision/Recall**: Métricas balanceadas para aplicaciones médicas
- **Matriz de confusión**: Análisis detallado de TP, TN, FP, FN


