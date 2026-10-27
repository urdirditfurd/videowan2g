# videowan2g

Pipeline headless qui transforme chaque scène de `scenes.json` en vidéo lofi style Ghibli (capybara), puis pousse le `.mp4` sur ce dépôt.

```bash
npm ci
cp .env.example .env
node automation.js --self-test
node automation.js
```

Renseignez `AGNES_API_KEY` et `GITHUB_TOKEN` dans `.env`. Ce fichier reste local. `--self-test` ne contacte ni Agnes ni GitHub.
