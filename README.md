# videowan2g

Pipeline headless qui transforme une courte description en vidéo lofi style Ghibli, avec une capybara, puis pousse le `.mp4` sur ce dépôt.

```bash
npm install
cp .env.example .env
node automation.js --self-test
node automation.js
```

Renseignez `AGNES_API_KEY`, `GITHUB_TOKEN` et `GITHUB_REPO_URL` dans `.env`. Les scènes lues sont dans `scenes.json`. Les vidéos arrivent dans `generated_videos/`.

`node automation.js --self-test` ne contacte aucune API et ne pousse rien.
