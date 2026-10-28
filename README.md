# videowan2g

Génération headless de vidéos lofi style Ghibli (mascotte capybara) via l’API Agnes, puis publication des fichiers `.mp4` sur ce dépôt.

```bash
npm install
cp .env.example .env
node automation.js --self-test
node automation.js
```

Renseignez `AGNES_API_KEY`, `GITHUB_TOKEN` et `GIT_REMOTE_URL` dans `.env`. Les scènes sont lues dans `scenes.json`. Les vidéos sont enregistrées dans `generated_videos/`.
