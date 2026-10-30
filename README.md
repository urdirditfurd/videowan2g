# videowan2g

Génération headless de vidéos lofi style Ghibli (mascotte capybara) via l’API Agnes, puis publication du `.mp4` sur ce dépôt.

```bash
npm install
cp .env.example .env
node automation.js --self-test
node automation.js
```

Renseignez `AGNES_API_KEY`, `GITHUB_TOKEN` et `GITHUB_REPO_URL` dans `.env`. Le fichier `.env` n’est pas versionné. `node automation.js --self-test` ne contacte pas l’API Agnes.
