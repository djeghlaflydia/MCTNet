# MCTNet & ECMTNet: Deep Learning for Crop Classification

Ce projet implémente et améliore l'architecture **MCTNet** (Multi-scale CNN-Transformer Network) pour la classification des cultures par pixel en utilisant des séries temporelles d'images Sentinel-2 (36 dates).

---

## 🚀 Présentation du Projet

L'objectif de ce travail est de reproduire les résultats du papier de recherche *"A lightweight CNN-Transformer network for pixel-based crop mapping"* (Wang et al., 2024) et de proposer une version améliorée nommée **ECMTNet**.

Le projet traite deux zones d'étude majeures :
*   **Arkansas** : 5 classes (Corn, Cotton, Rice, Soybeans, Others).
*   **California** : 6 classes (Grapes, Rice, Alfalfa, Almonds, Pistachios, Others).

---

## 🧠 Architecture des Modèles

### 1. MCTNet (Modèle Original)
*   **Hiérarchie à 3 étages** : Réduction progressive de la résolution temporelle (36 → 18 → 9 dates).
*   **ALPE (Adaptive Learned Positional Encoding)** : Gestion intelligente des dates manquantes et encodage temporel.
*   **Double Branche** : CNN pour les motifs locaux et Transformer pour les dépendances à long terme.

### 2. ECMTNet (Notre Modèle Amélioré)
*   **Gated Fusion** : Remplace la concaténation simple par une porte logique apprise qui pondère l'importance du CNN par rapport au Transformer.
*   **Phenology Attention** : Module d'attention temporelle focalisé sur les stades phénologiques critiques (floraison, maturité).
*   **Cross-Scale Fusion** : Agrégation des caractéristiques de tous les étages (multi-résolution) pour une classification plus robuste.

---

## 📂 Structure du Répertoire

*   `01_data_exploration.py` : Analyse des données et profils spectraux.
*   `02_preprocessing.py` : Nettoyage, normalisation et création des tenseurs PyTorch.
*   `03_train.py` : Entraînement du modèle MCTNet de base.
*   `04_evaluate.py` : Évaluation et comparaison avec les baselines du papier.
*   `07_ablation_study.py` : Analyse de l'impact des variables environnementales (Sol, Climat).
*   `09_train_ecmtnet.py` : Entraînement comparatif MCTNet vs ECMTNet.
*   `10_analyze_part3.py` : Visualisation des performances de la Partie 3.
*   `models/` : Contient les implémentations PyTorch de `mctnet.py` et `ecmtnet.py`.

---

## 🛠️ Installation et Utilisation

### Prérequis
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

### Étapes d'exécution

1.  **Préparer les données** :
    ```bash
    python 02_preprocessing.py
    ```

2.  **Entraîner le modèle amélioré (Partie 3)** :
    ```bash
    # Par défaut, traite Arkansas et California séquentiellement
    python 09_train_ecmtnet.py
    ```

3.  **Analyser et comparer** :
    ```bash
    python 10_analyze_part3.py
    ```

---

## 📊 Résultats
Les résultats (courbes d'apprentissage, matrices de confusion, rapports de classification) sont automatiquement sauvegardés dans le dossier `./results/`.

---
**USTHB - M1 SII 2025/2026**  
*Projet de Deep Learning pour la Classification des Cultures.*
