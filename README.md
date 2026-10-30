# videowan2g

Vidéos lofi style Studio Ghibli, avec une capybara, générées par l'API Agnes AI puis publiées sur ce dépôt.

## Installation

```bash
npm install
cp .env.example .env
```

Renseignez `AGNES_API_KEY` et `GITHUB_TOKEN` dans `.env`. Le dépôt accepté est `https://github.com/urdirditfurd/videowan2g`.

## Lancement

```bash
node automation.js --self-test
node automation.js
```

`--self-test` vérifie le parseur, la file d'attente, le filtre de téléchargement et l'isolation Git sans appeler l'API. Le mode normal lit `scenes.json`, écrit les `.mp4` dans `generated_videos/`, puis fait `git add -f`, `git commit` et `git push`.
