# DARTS: Differentiable Architecture Search - Pipeline Optimizado

Este repositorio contiene una implementación moderna y optimizada del algoritmo **DARTS** (*Differentiable Architecture Search*) para la búsqueda automática de arquitecturas convolucionales en CIFAR-10. Incluye dos scripts principales:

- `search_darts.py`: búsqueda de la mejor arquitectura (celda) mediante optimización bi‑nivel.
- `eval_darts.py`: evaluación de la arquitectura encontrada entrenando una red grande desde cero.

El código incorpora múltiples mejoras de rendimiento respecto a la implementación original (PyTorch 0.3): **preprocesamiento offline**, **mixed precision (AMP)**, **gradient accumulation**, **`torch.compile`**, **validación espaciada** y **DataLoaders optimizados**.

---

##  Requisitos

- Python ≥ 3.8
- PyTorch ≥ 2.0 (recomendado para `torch.compile`)
- torchvision
- numpy, matplotlib, seaborn, scikit‑learn, scipy
- tqdm (opcional)

Instalación rápida:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn scipy