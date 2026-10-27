# videowan2g

Génération headless de vidéos lofi (style Ghibli, mascotte capybara) via l’API Agnes, puis publication du `.mp4` sur ce dépôt.

## Installation

```bash
npm install
cp .env.example .env
```

Renseignez `AGNES_API_KEY` et `GITHUB_TOKEN` dans `.env`. Ne commitez jamais ce fichier.

## Lancement

```bash
node automation.js
```

Le script lit `scenes.json`, enchaîne expansion texte, image 16:9 et vidéo `keyframe`, télécharge le mp4 dans `generated_videos/`, puis fait `git add`, `git commit` et `git push` vers `GITHUB_REPO_URL`.

Si Agnes répond `video_queue_full`, la tâche attend 60 s, puis 120 s, puis 300 s (avec un léger aléa, plafond 360 s) sans s’arrêter et sans épuiser le budget de tentatives. Un HTTP 429 reprend aussi, avec le backoff du studio d’origine.

Le mp4 n’est publié que s’il provient d’une URL publique (les redirections vers une adresse locale ou le service de métadonnées sont refusées) et si son en-tête est bien celui d’un MP4 ou d’un WebM.

## Vérification hors réseau

```bash
node automation.js --self-test
```
