# videowan2g

Boucles vidéo de 3 à 5 secondes, style **lofi / Ghibli**, avec une mascotte **capybara**. Le dépôt prépare un prompt stable, lance [WanGP](https://github.com/deepbeepmeep/Wan2GP) (Wan2GP), puis ferme la boucle et réchauffe l'image avec FFmpeg.

Ce projet utilise WanGP. L'usage de son API est soumis aux conditions du projet WanGP.

## Ce que WanGP permet

Relevé sur la documentation et le handler Wan du dépôt `deepbeepmeep/Wan2GP` :

| Besoin | Réglage retenu |
| --- | --- |
| Texte vers vidéo, peu de VRAM | `model_type: t2v_1.3B`, drapeau `--t2v-1-3B`. Environ 6 Go. Le 14B demande plutôt 12 Go. |
| Image vers vidéo | `fun_inp_1.3B`, drapeau `--i2v-1-3B`, dès qu'une image de départ est fournie. |
| Durée | 16 i/s pour cette famille. Nombre d'images valide : 5, 9, 13… (minimum 5, pas de 4). **81 images = 5,06 s**. |
| VRAM | Profil **4** : le modèle monte sur le GPU par blocs. Profil 5 si la RAM est juste elle aussi. Attention `sdpa` pour les GTX 10XX comme pour les RTX récentes. |
| Lancement sans interface | `python wgp.py --process reglages.json --output-dir ...` |
| API Python | `shared.api.init(...).submit_task(reglages)`, dans le même interpréteur que le conda WanGP. |

Le fichier de réglages suit les clés documentées : `model_type`, `prompt`, `negative_prompt`, `resolution`, `video_length`, `force_fps`, `num_inference_steps`, `guidance_scale`, `seed`, `override_profile`.

Les résolutions par défaut restent au voisinage du 480p du modèle 1.3B, sur la grille de 16 pixels du VAE : **480×832** (9:16) et **624×624** (1:1).

## Esthétique intégrée

Chaque prompt reçoit, depuis le code, les mêmes contraintes :

- un seul capybara (fourrure châtain, museau crème, pull crème, regard mi-clos) ;
- un fond qui vit tout seul : pluie, herbe, vapeur, nuages, rideau ;
- des micro-mouvements : respiration, clignement, petite gorgée ;
- un plan fixe, sans secousse ;
- une palette pastel et une lumière chaude ;
- une consigne de boucle : la fin revient vers le début.

Le seed choisit une variation du thème (plaid, théière, carnet…) et fixe le bruit du modèle. Deux seeds du même thème ne donnent pas le même plan. La mascotte, elle, ne change pas.

## Arborescence

```
config/themes.yaml        thèmes, mouvement lent, variations
config/pipeline.yaml      modèle, profil VRAM, résolution, grain
src/prompt_builder.py     prompt final
src/wan2gp_runner.py      réglages JSON et processus WanGP
src/post_processor.py     boucle, chaleur, grain, audio
src/pipeline.py           enchaînement
main.py                   commande
output/                   vidéos et manifestes
```

## Installation de WanGP

Carte NVIDIA, avec les consignes du dépôt WanGP. Résumé pour une RTX 20XX à 50XX :

```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

GTX 10XX : le guide WanGP utilise Python 3.10.9 et PyTorch 2.7.1 (cu128). Dans `config/pipeline.yaml`, mettre `fp16: true` et laisser `attention: sdpa`.

La première génération télécharge le poids `wan2.1_text2video_1.3B`. Prévoir plusieurs gigaoctets de disque.

Vérifier l'identifiant du modèle dans l'interface (bouton d'export des réglages). S'il diffère de `t2v_1.3B`, le reporter dans `wan2gp.model_type`.

## Installation de ce dépôt

Dans le même environnement conda, pour que le backend `api` voie PyTorch et `shared.api` :

```bash
conda activate wan2gp
cd /chemin/vers/videowan2g
pip install -r requirements.txt
export WAN2GP_ROOT=/chemin/vers/Wan2GP
```

`ffmpeg` et `ffprobe` servent à la boucle. Ils tournent sur le CPU. Les laisser dans le PATH : `apt install ffmpeg` sous Debian/Ubuntu, build officiel avec PATH sous Windows.

`WAN2GP_PYTHON` peut pointer vers le Python du conda si `main.py` est lancé depuis un autre interpréteur. Le backend `subprocess` appelle alors cet interpréteur. Le backend `api` doit, lui, être lancé par le Python du conda.

## Utilisation

```bash
python main.py --list
python main.py --theme rainy_window --dry-run
python main.py --theme cozy_cafe
python main.py --theme sunset_train --seed 11 --aspect 1:1
python main.py --theme night_train --image /chemin/still.png
python main.py --theme rainy_window --audio /chemin/lit-cc0.wav
```

`--dry-run` écrit le prompt et le JSON WanGP dans `output/`, sans toucher au GPU.

Une génération réelle produit :

- `output/capybara_<theme>_<seed>.mp4`
- `output/capybara_<theme>_<seed>.json` (prompt, seed, commande)
- `output/raw/` la vidéo WanGP avant finition
- `output/logs/` le journal du processus

Thèmes livrés : `rainy_window`, `sunset_train`, `night_train`, `cozy_cafe`, `golden_meadow`, `snowy_bookshop`, `forest_stream`, `starry_rooftop`, `lakeside_boat`, `rainy_balcony`.

Pour allonger la boucle, augmenter `wan2gp.frames` en restant sur la grille 5 + 4k (le script aligne tout seul vers le palier supérieur). Le 1.3B est à l'aise jusqu'à 81 images en 480×832. Au-delà, la VRAM et le temps montent ; un avertissement est écrit dans le journal.

## Boucle et rendu lofi

FFmpeg coupe un court amorce, puis fond la fin du plan vers cette amorce. La première image et la dernière image du fichier livré sont la même image source : en lecture en boucle, la jointure disparaît. Le fondu par défaut dure 0,35 s, donc une source de 5,06 s sort autour de 4,7 s.

Le même encodage pose une chaleur pastel (saturation un peu basse, ombres bleues retirées, hautes lumières dorées) et un grain léger. `post.grain: 0` retire le grain. `post.warm_grade: false` garde les couleurs du modèle.

## Audio

Le script ne télécharge aucune musique. Déposer un fichier dont la licence le permet (CC0, domaine public, ou CC-BY avec la mention demandée par l'auteur), puis :

```yaml
audio:
  enabled: true
  path: "/chemin/vers/boucle-cc0.wav"
  volume: 0.22
```

Ou passer `--audio` pour un seul film. La piste est mise en boucle sur la durée de la vidéo, avec une entrée et une sortie en fondu.

## VRAM

Le processus headless est le mode par défaut : à la fin de `wgp.py`, la mémoire GPU est rendue. Le profil 4 et la quantification déjà active dans WanGP sont ce qui fait tenir le 1.3B vers 6 Go.

Si la carte sature encore :

1. rester en `480x832` et 81 images ;
2. passer `profile: 5` ;
3. fermer les autres programmes qui occupent le GPU (un navigateur avec accélération compte) ;
4. garder `attention: sdpa` sur une GTX 10XX ;
5. éviter `t2v` / `i2v` 14B sous 12 Go.

`backend: api` garde le modèle en mémoire entre deux films. Utile pour une série, à lancer avec le Python du conda :

```bash
python main.py --theme cozy_cafe --backend api
```

## Tests

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
```

Les tests couvrent le prompt, la grille d'images, le JSON headless et la boucle FFmpeg. WanGP reste en dehors de cette suite.

## Dépannage

- `WAN2GP_ROOT est vide` : exporter le chemin du clone, ou remplir `wan2gp.root`.
- `wgp.py introuvable` : le chemin pointe à côté du dépôt WanGP.
- WanGP s'arrête tout de suite : lire `output/logs/`. Un `model_type` inconnu de cette version se corrige via l'export de réglages de l'interface.
- Image de départ ignorée : le chemin doit être lisible par le processus WanGP. Le script envoie un chemin absolu et `image_prompt_type: S`.
- La boucle saute encore : baisser `post.crossfade_seconds` reste au-dessus de 0,2 s, ou générer à nouveau. Le mouvement très lent est aussi porté par le prompt.
