# Performance

Mesures d'inférence par fenêtre de 3 s, backend `onnxruntime-web` WASM, **SIMD activé,
`numThreads = 1`**, contre le bundle de production (`dist/`).

Reproduire :

```bash
pnpm bench            # 3 passes sur soundscape.wav (40 fenêtres chacune)
pnpm bench --runs=5
```

## Ce que j'ai pu mesurer, et ce que je n'ai pas pu

L'environnement d'exécution de ce travail est **Linux x86_64, 4 cœurs**. Les chiffres demandés sur
**Mac Apple Silicon, Chrome et Safari, n'ont donc pas pu être produits** : il n'y a ni matériel Apple
ni Safari ici. Ce qui suit est mesuré sur Chromium/Linux ; `pnpm bench` est prévu pour être relancé
tel quel sur un Mac et remplir les lignes manquantes.

| Plateforme | Navigateur | Médiane / fenêtre | p95 | Temps réel |
|---|---|---|---|---|
| Linux x86_64, 4 cœurs | Chromium 141 headless | **103,6 ms** | 136,9 ms | **29×** |
| macOS Apple Silicon | Chrome | _à mesurer_ | | |
| macOS Apple Silicon | Safari | _à mesurer_ | | |

Mesuré sur 120 fenêtres (3 passes × 40), `soundscape.wav`. Min 96,7 ms, max 336,8 ms — le max est la
première fenêtre, avant que le runtime n'ait chauffé.

Autres coûts sur la même machine :

| Étape | Temps |
|---|---|
| Décodage + rééchantillonnage de 120 s d'audio | 204 ms |
| Chargement du modèle (52 Mo, réseau local, cache vide) | 1 535 ms |
| **Fichier de 2 minutes, bout en bout** | **~4 s** |

`docs/bench-latest.json` est écrit par `pnpm bench` et contient la dernière mesure locale.

29× temps réel en mono-thread laisse une marge confortable pour le mode micro de P3, qui n'a besoin
que de 1×.

## Lecture des chiffres

- **Médiane par fenêtre** : coût d'une inférence sur 3 s d'audio. C'est le seul chiffre qui compte
  pour la fluidité du flux de détections.
- **Facteur temps réel** = `3000 / médiane_ms`. Au-dessus de 1, on analyse plus vite que le temps qui
  passe — condition nécessaire au mode micro de P3.
- **`modelLoadMs`** : premier chargement, réseau local, cache vide. En conditions réelles c'est
  dominé par le téléchargement des 52 Mo ; ensuite la Cache API rend le coût négligeable.

## Pourquoi mono-thread

Le multi-thread d'ORT exige `SharedArrayBuffer`, donc les en-têtes COOP/COEP, qui cassent les iframes
tiers et imposent une configuration serveur qu'un site statique n'a pas. On mesure d'abord en
mono-thread ; si les chiffres l'imposent, le multi-thread se réévalue avec ses contraintes de
déploiement en pleine connaissance de cause.

Le SIMD, lui, est activé : c'est le gain principal et il ne coûte aucun en-tête.

## Note sur le coût du front-end mel

La conversion replie la STFT + le banc de filtres mel en une seule convolution 1-D (voir README).
Ce n'est pas qu'une question de compatibilité : la DFT dense équivalente représenterait environ
2,7 GFLOP par fenêtre, contre 0,826 GFLOP pour le modèle entier. Le repli garde donc l'inférence
navigateur dans un ordre de grandeur utilisable.
