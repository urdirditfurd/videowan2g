# videowan2g

Génération headless de vidéos lofi (aquarelle façon Studio Ghibli, mascotte capybara) via l'API Agnes, puis publication des fichiers `.mp4` sur ce dépôt.

Le script lit `scenes.json`, enrichit chaque scène, crée une image 16:9, lance une vidéo en mode `keyframe`, attend la fin du rendu, enregistre le mp4 dans `generated_videos/`, puis fait `git add -f`, `git commit` et `git push`. Les mp4 restent ignorés par Git tant que le script ne publie pas celui qu'il vient d'écrire.

Si la file vidéo Agnes répond `video_queue_full` ou 429, la tâche attend (60 s, puis 120 s, puis 300 s, avec un peu de hasard) sans s'arrêter.

## Installation

```bash
npm install
cp .env.example .env
```

Renseignez `AGNES_API_KEY`, `GITHUB_TOKEN` et `GITHUB_REPO_URL` dans `.env`. Le dépôt accepté est `https://github.com/urdirditfurd/videowan2g`.

## Lancement

```bash
node automation.js --self-test
node automation.js
```

`--self-test` vérifie le parseur, les URL, la file d'attente et l'isolation git, sans appel réseau et sans clé API.
