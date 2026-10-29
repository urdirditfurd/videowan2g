# videowan2g

Pipeline headless : chaque scène de `scenes.json` est enrichie en ambiance lofi style Ghibli avec une capybara, puis une image 16:9 et une vidéo sont générées via Agnes AI. Le mp4 est enregistré dans `generated_videos/` et poussé sur ce dépôt.

```bash
npm install
cp .env.example .env
node automation.js --self-test
node automation.js
```

Renseignez `AGNES_API_KEY` et `GITHUB_TOKEN` dans `.env` avant le lancement réel. `--self-test` ne contacte aucune API et ne pousse rien.
