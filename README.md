# videowan2g

Génération headless de vidéos lofi style Ghibli, avec une capybara, via l’API Agnes. Chaque scène de `scenes.json` est enrichie, illustrée, animée, enregistrée dans `generated_videos/`, puis publiée sur ce dépôt.

## Installation

```bash
npm install
cp .env.example .env
```

Renseignez `AGNES_API_KEY` et `GITHUB_TOKEN` dans `.env`. Le dépôt accepté est uniquement `https://github.com/urdirditfurd/videowan2g`.

## Lancement

```bash
node automation.js --self-test
node automation.js
```

`--self-test` vérifie la file d’attente, les téléchargements et l’isolation des secrets sans appeler l’API. `node automation.js` enchaîne les scènes : expansion, image 16:9, vidéo keyframe, téléchargement `.mp4`, puis `git add -f`, commit et push.
