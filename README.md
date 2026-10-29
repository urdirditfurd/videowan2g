# videowan2g

Pipeline headless qui transforme chaque scène de `scenes.json` en vidéo lofi style Ghibli avec une capybara, via l'API Agnes, puis pousse le `.mp4` sur ce dépôt.

```bash
npm install
cp .env.example .env
node automation.js --self-test
node automation.js
```

Renseignez `AGNES_API_KEY`, `GITHUB_TOKEN` et `GIT_REPO_URL` dans `.env`. Le fichier `.env` reste local. `node automation.js --self-test` ne contacte ni Agnes ni GitHub.
