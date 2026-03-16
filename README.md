# Landing Pad – Safe / Not Safe (YOLO)

## Objectif
Ce projet entraîne un modèle de détection d’objets (YOLO/Ultralytics) pour détecter une **landing pad** sur une image, puis décide si la scène est **SAFE** ou **NOT SAFE** pour l’atterrissage.

La décision SAFE est prise si le modèle détecte au moins une bounding box de landing pad avec :
- une confiance `conf >= tau_conf`
- une aire relative `area_ratio >= tau_area` (aire de la box / aire de l’image)

## Organisation du dépôt
```
.
├── LANDING PAD.v2i.yolov8/
│   ├── runs/
│   │  ├── detect/...
│   │  └── postproc/
│   │    ├── best_thresholds.json
│   │    ├── tuning_results.csv
│   │    ├── test_predictions.csv
│   │    ├── live_log.csv
│   │    └── live_demo.(mp4/avi)          # Dataset YOLO (Roboflow export)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml
│   ├── Projet_HAFAIEDH_FELIX.ipynb
│   └── README.md
```

## Prérequis
- Python 3.10+ (3.12 OK)
- Packages :
  - `ultralytics`
  - `numpy`
  - `pandas`
  - `pyyaml`
  - `opencv-python` (uniquement pour la démo live + enregistrement vidéo)

Installation :
```bash
pip install ultralytics numpy pandas pyyaml opencv-python
```

## Dataset
Le dataset est au format YOLO :
- une image `xxx.jpg` correspond à un label `xxx.txt` (même nom de base)
- un label YOLO contient : `class_id x_center y_center width height` (valeurs normalisées entre 0 et 1)
- pour une image **sans pad**, le fichier label est **vide** 

Classe utilisée :
- `0 = landing_pad`

## Reproduire les résultats (pipeline)
Le pipeline se fait dans `Projet_HAFAIEDH_FELIX.ipynb` :

1. **Vérifier les chemins du dataset**
   - Ouvrir `data.yaml`
   - Si les chemins relatifs posent problème sur Windows, générer un `data_abs.yaml` avec chemins absolus.

2. **Entraîner le modèle YOLO (transfer learning)**
   - Le notebook lance l’entraînement et génère un dossier `runs/detect/...`
   - Le meilleur modèle est sauvegardé en `best.pt`

3. **Choisir les seuils (tuning) sur la validation**
   - Recherche des meilleurs `tau_conf` et `tau_area` sur le split `valid`
   - Sorties :
     - `runs/postproc/best_thresholds.json`
     - `runs/postproc/tuning_results.csv`

4. **Évaluer sur le test (seuils figés)**
   - Application des seuils choisis sur `test`
   - Sorties :
     - `runs/postproc/test_predictions.csv` (GT vs prédiction + scores)

5. **Démo live webcam + log + enregistrement vidéo**
   - Affichage de la meilleure bounding box, décision SAFE/NOT SAFE
   - Sorties :
     - `runs/postproc/live_log.csv`
     - `runs/postproc/live_demo.mp4` (ou `.avi`)

## Fichiers de sortie importants
- `runs/detect/.../weights/best.pt` : modèle entraîné
- `runs/postproc/best_thresholds.json` : seuils finaux (validation)
- `runs/postproc/test_predictions.csv` : résultats test détaillés
- `runs/postproc/live_log.csv` : log de la démo live
- `runs/postproc/live_demo.mp4` : vidéo de démonstration

## Notes
- Les seuils sont choisis **uniquement** sur la validation puis figés.
- En cas de comportement trop “conservateur” (beaucoup de FN), tester une grille plus large ou réduire `tau_area`.
