# whosyourbirdy

Identification d'oiseaux au chant, **entièrement dans le navigateur**, au-dessus du modèle
[BirdNET v2.4](https://github.com/birdnet-team/BirdNET-Analyzer) du Cornell Lab of Ornithology et de
la TU Chemnitz.

Aucun backend, aucun upload : le fichier audio ne quitte jamais la machine. Le modèle (~52 Mo) est
téléchargé une fois, mis en cache, et l'inférence tourne en WebAssembly dans un Web Worker.

> **Powered by BirdNET** — K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of
> Ornithology & Chemnitz University of Technology.
> Modèle sous **CC BY-NC-SA 4.0** : usage **non commercial**, partage à l'identique, attribution
> obligatoire. Ce projet est non commercial.

## État

**P0 et P1 livrés** : le pipeline, la preuve de parité numérique avec l'implémentation officielle, et
une interface minimale fonctionnelle — dépôt d'un fichier, détections triées (espèce, score,
timecode), seuil de confiance réglable, lecture du segment au clic.

Le spectrogramme et la timeline arrivent en P2.

> **Une limite connue dépasse le contrat de 1 × 10⁻³.** Sur la **dernière fenêtre** d'un fichier dont
> la durée n'est pas un multiple de 3 s (donc zero-paddée), l'écart de score atteint 1,7 × 10⁻².
> Toutes les autres fenêtres restent à ≤ 9,4 × 10⁻⁵. Aucune détection ne change, sur 35 fenêtres
> paddées testées. Cause identifiée et bornée : voir [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Démarrage

```bash
pnpm install

# Environnement Python pour la construction du modèle et le test de parité
python3 -m venv .venv
.venv/bin/pip install numpy==1.26.4 tensorflow-cpu==2.15.1 tf2onnx==1.16.1 \
                      onnx==1.16.2 onnxruntime==1.19.2 soundfile librosa resampy

pnpm model:build   # récupère les artefacts officiels et produit public/models/
pnpm parity        # prouve que le pipeline reproduit BirdNET
pnpm smoke         # pilote l'interface dans un vrai Chromium
pnpm dev
```

Les poids ne sont **pas** dans le dépôt. `pnpm model:build` les régénère depuis les artefacts
officiels, ce qui garde le dépôt léger et évite de redistribuer un modèle sous licence NC.

## La chaîne audio, et pourquoi chaque choix

### 1. Décodage — `decodeAudioData`

Le navigateur décode wav, mp3, flac, m4a, ogg selon son moteur. On ne réimplémente aucun décodeur.

### 2. Downmix mono par moyenne des canaux

BirdNET analyse du mono. La moyenne (et non la sélection du canal gauche) préserve les sources
présentes uniquement sur un canal.

### 3. Rééchantillonnage à 48 kHz

Le modèle a été entraîné à 48 kHz. Le rééchantillonnage est **délégué au navigateur** : écrire notre
propre rééchantillonneur ne nous rapprocherait pas de `resampy` (celui de BirdNET), ça déplacerait
juste l'écart.

Détail qui compte, vérifié à l'exécution : **`decodeAudioData` rééchantillonne vers la fréquence du
contexte de décodage**. On décode donc délibérément dans un `OfflineAudioContext` à 48 kHz — c'est
aussi ce que fait la démo navigateur officielle de BirdNET, et pour la même raison : le contexte par
défaut à 44,1 kHz sous-échantillonnerait silencieusement chaque fichier. Sous Chromium et Firefox,
le rééchantillonnage a donc déjà eu lieu à la sortie du décodeur.

L'étape `resampleTo48k` explicite reste là comme **repli réel** : Safari a historiquement renvoyé la
fréquence propre du fichier plutôt que celle du contexte. Ce n'est pas du code mort.

Un fichier déjà en 48 kHz traverse la chaîne **sans être altéré** — cas le plus fréquent en
enregistrement de terrain, et c'est le régime du niveau B (9,4 × 10⁻⁵).

L'écart résiduel sur les fichiers à convertir est **mesuré** par le niveau C, jamais compensé.

### 4. Fenêtrage : 144 000 échantillons, hop de 144 000

Soit exactement 3 s, la taille d'entrée du modèle. Le paramètre `overlap` (0 à 2,9 s, comme
BirdNET-Analyzer) réduit le hop. La dernière fenêtre est **zero-paddée** et conservée, pas jetée :
une détection dans les dernières secondes ne doit pas disparaître.

### 5. Ce qu'on ne fait surtout pas

**Pas de normalisation d'amplitude** (ni peak, ni RMS), **pas de filtre** passe-haut ou passe-bas,
**pas d'aller-retour en int16**.

Ce n'est pas une préférence de style : le modèle **normalise déjà lui-même**. Sa couche mel commence
par un min-max par fenêtre de 3 s vers `[-1, 1]` :

```
x = x - min(x);  x = x / (max(x) + 1e-6);  x = (x - 0.5) * 2
```

Normaliser en amont change les statistiques de fenêtre sur lesquelles le modèle a été entraîné et
dégrade les scores. `decodeAudioData` sort déjà du float dans `[-1, 1]` : c'est exactement le domaine
attendu.

### 6. Inférence dans un Web Worker

6 522 classes × N fenêtres bloquerait le thread principal. Le worker émet **un message par fenêtre**,
donc les détections arrivent en flux et l'interface peut afficher le front d'analyse progresser.

Le PCM est **transféré** (pas copié) vers le worker : un fichier de 2 minutes se déplace comme un
pointeur, pas comme 23 Mo.

### 7. Sortie : des logits, pas des probabilités

Le modèle renvoie des logits. Le score s'obtient par le *flat sigmoid* de BirdNET :

```
y = -sensitivity * clip(logit + (bias - 1) * 10, ±15)
score = y >= 0 ? e^-|y| / (1 + e^-|y|) : 1 / (1 + e^-|y|)
```

Aux valeurs par défaut (`sensitivity = 1.0`) cela se réduit à la sigmoïde logistique simple — mais le
**clip à ±15** compte, et la sensibilité n'est réglable que si l'activation reste hors du graphe.
C'est pourquoi on exporte les logits.

**Sigmoïde et non softmax** : c'est du multi-label. Plusieurs espèces peuvent chanter sur la même
fenêtre, les scores ne somment pas à 1.

## D'où vient le modèle

Les builds ONNX publiés sur Hugging Face et les poids servis par Zenodo sont inaccessibles depuis
certains réseaux (politique d'egress). On reconstruit donc le modèle depuis des artefacts atteignables
en git, ce qui a l'avantage d'être vérifiable :

| Élément | Source (tag `v1.5.1` de `birdnet-team/BirdNET-Analyzer`) |
|---|---|
| Topologie | `checkpoints/V2.4/..._Model_TFJS/static/model/model.json` (config Keras) |
| Poids | les 13 shards TFJS associés (226 tenseurs nommés) |
| **Référence de vérification** | `checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite` |
| Labels fr / en | `labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_{fr,en}.txt` (6 522 lignes) |
| Modèle géo (MData, pour P4) | `checkpoints/V2.4/..._MData_Model_V2_FP16.tflite` — vérifié : `[1,3]` (lat, lon, semaine) → `[1,6522]`, ops triviales, aucune chirurgie nécessaire |
| Fixture audio | `example/soundscape.wav` (120 s, mono, 48 kHz) |

### La seule modification apportée au graphe

BirdNET calcule ses deux mel-spectrogrammes **dans le graphe**, via `tf.signal.stft`. Deux détails
rendent ce graphe inconvertible tel quel :

- BirdNET utilise la **partie réelle** de la STFT, pas son module (`tf.cast(spec, 'float32')` sur un
  complexe). Le convertisseur `tf2onnx` ne sait traiter que le motif `RFFT2D → ComplexAbs`.
- L'opérateur `DFT` d'ONNX n'est pas supporté de façon fiable par le backend WASM d'ONNX Runtime Web.

Or trois opérations consécutives sont **linéaires** dans le signal fenêtré : la fenêtre de Hann, la
DFT réelle, et le banc de filtres mel. Leur composition est donc une seule matrice constante :

```
mel[t,m] = Σ_n frames[t,n] · ( hann[n] · Σ_k cos(2πnk/N) · MB[k,m] )
```

et « faire glisser une matrice fixe sur un signal avec un pas » est exactement une **convolution 1-D
stridée**. Tout le front-end STFT + mel devient donc **une seule `Conv1D`** dont le noyau est dérivé
de la fenêtre et du banc de filtres du modèle lui-même.

Ce n'est pas une réimplémentation du mel-spectrogramme : c'est le même opérateur linéaire,
matérialisé. Rien n'est re-dérivé ni re-réglé — les bancs de filtres, les tailles de trame et le
`magnitude_scaling` viennent tous de la config et du checkpoint officiels. Bénéfice secondaire : on
remplace ~1 GFLOP de DFT dense par une convolution à 96 canaux.

`scripts/convert_to_onnx.py` **assert** l'équivalence à chaque étape et refuse d'écrire un modèle qui
ne correspond pas. Voir `scripts/birdnet_mel.py` pour la dérivation complète.

#### Ce que ce repli coûte

L'algèbre est exacte, la stabilité numérique ne l'est pas. Une FFT accumule en `log₂(2048) = 11`
étages (erreur en `O(√log N)`) ; un produit scalaire direct accumule 2048 termes (erreur en `O(√N)`).
Sur de l'audio réel c'est invisible — 6,8 × 10⁻⁵ d'écart de score. Sur une trame quasi silencieuse,
où la vraie valeur mel vaut ~0 et où le résultat *est* le résidu d'arrondi, le repli est ~4,6× moins
bien conditionné que la FFT qu'il remplace.

C'est l'origine exacte de la limite documentée dans
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) sur la fenêtre finale zero-paddée. Le compromis est
assumé et mesuré, pas ignoré : sans ce repli il n'y a pas d'inférence navigateur du tout.

## Le test de parité

```bash
pnpm parity            # les trois niveaux
pnpm parity --only=A   # un seul
```

Cinq niveaux, ordonnés pour qu'un échec désigne **une** cause :

| Niveau | Ce qui est comparé | Verdict |
|---|---|---|
| **A** | PCM identique → TFLite officiel (Python) **vs** notre ONNX sous ORT WASM. Isole le modèle. | doit passer |
| **A′** | 35 fenêtres zero-paddées : 5 positions dans le fichier × 7 longueurs de queue. | borne documentée |
| **B** | Le même fichier → référence Python **vs** vrai Chromium exécutant `lib/birdnet` (Web Audio, worker, sigmoïde). Isole la chaîne audio. | doit passer |
| **C** | Fichier en 44,1 kHz : les deux côtés doivent rééchantillonner, avec des rééchantillonneurs différents. | mesuré, non imposé |
| **D** | Entrées dégénérées (silence total, continu, saturation, tons purs) envoyées directement aux deux implémentations, sans chaîne audio. | doit passer |

Le niveau A′ existe parce que `soundscape.wav` fait exactement 40 fenêtres pleines : sans lui, le
chemin de zero-padding — donc toute erreur d'un cran dans le fenêtrage — ne serait jamais exercé. Il
balaie **position et longueur de queue**, parce que l'erreur dépend fortement des deux : une seule
troncature bien choisie donne 2,4 × 10⁻⁴ et une fausse impression de sécurité, là où le balayage
complet révèle 1,7 × 10⁻².

Le niveau D existe parce que le front-end mel est le plus fragile là où le signal est pauvre : le
modèle divise par `max(x) + 1e-6` puis élève à une puissance fractionnaire. Le cas « fenêtre
entièrement nulle » n'est pas théorique — tout fichier dont la durée n'est pas un multiple de 3 s en
produit une partielle, et un fichier se terminant par du silence en produit une complète.

Le niveau D existe aussi parce que le harnais doit exercer les **valeurs par défaut réelles** :
l'analyseur y tourne avec `excludeNonEvents: true`, donc le worker emprunte sa branche
`allowedClasses` — celle que prend tout appel réel. Une version antérieure passait `false` et
laissait cette branche entièrement non testée.

### Ce sur quoi le harnais statue, et ce qu'il se contente de rapporter

Le critère est **`max|Δscore| ≤ 1e-3`** et **zéro divergence de détection au seuil de 0,25**. Ce sont
les deux seules propriétés observables par un utilisateur.

`max|Δlogit|` est rapporté comme diagnostic, avec une borne volontairement lâche (0,1) qui ne sert
qu'à détecter une casse franche. Raison : sur une fenêtre à moitié silencieuse, la plupart des bandes
mel tendent vers zéro, et le `pow(x, 1/(1+e^s))` du modèle a une **dérivée infinie en 0** — il
amplifie donc le bruit float32 en logits (jusqu'à ~0,04) pour des classes que le modèle rejette de
toute façon. Sur le niveau A′, 4 495 classes dépassent 5 × 10⁻³ de Δlogit, toutes situées entre −18,5
et −4,1 de logit, soit des scores de 3 × 10⁻⁷ à 1,6 × 10⁻² : l'écart de score maximal reste
2,4 × 10⁻⁴ et **aucune** détection ne change.

Une vraie erreur de conversion (comparer des probabilités à des logits, intervertir les canaux mel,
inverser le spectrogramme) déplace les logits de 1 à 20 — deux ordres de grandeur au-dessus de la
borne. C'est ce qui la rend utile malgré sa largeur.

La référence est le tflite FP32 officiel passé dans l'interpréteur TensorFlow Lite, avec le
`flat_sigmoid` de BirdNET. C'est le chemin d'inférence du paquet `birdnet`, sans son downloader
Zenodo (injoignable ici).

Le niveau B ne se contente pas de comparer des tenseurs : il vérifie aussi que les détections
produites par `BirdNetAnalyzer` (worker, seuillage, mapping des labels) correspondent à ce que les
logits bruts impliquent. Une erreur dans le worker ne peut donc pas se cacher derrière un niveau B
vert.

## Performance

Voir `docs/PERF.md`. Mesures produites par `pnpm bench`.

## L'interface (P1)

Volontairement minimale : c'est P2 qui portera la direction visuelle, autour du spectrogramme.

Deux décisions valent d'être expliquées :

**Le seuil filtre en mémoire, il ne relance rien.** L'analyse tourne une fois à un plancher de 0,01 et
le curseur filtre le résultat déjà en RAM — ~140 ms au lieu de relancer 40 inférences. Ce plancher est
aussi la borne basse du curseur : proposer moins promettrait des résultats jamais calculés.

**La lecture passe par le fichier d'origine**, pas par le PCM décodé — celui-ci pèse des dizaines de
Mo et il est de toute façon transféré au worker. Un `objectURL` et un `<audio>` suffisent, et la
lecture s'arrête à la fin de la fenêtre de 3 s pour qu'on entende exactement ce que le modèle a noté.

### Annulation d'une analyse

Le worker **cède la main entre chaque fenêtre** (`setTimeout(0)`). Sans ça, `await inferWindow()`
n'atteint qu'un point de contrôle de microtâches, qui ne vide pas la file de messages du worker : un
`cancel` posté après le démarrage n'était jamais reçu, l'analyse allait à son terme, et ses résultats
arrivaient **sous le nom du fichier suivant**. Concrètement, on pouvait afficher 24 détections
d'oiseaux pour un fichier texte de 29 octets.

L'annulation restant par nature *best-effort* — le worker peut être au milieu d'une fenêtre — chaque
écriture d'état est en plus protégée par un jeton d'exécution : une analyse abandonnée ne peut pas
écrire dans l'interface. Les deux régressions sont couvertes par `pnpm smoke`.

`pnpm smoke` pilote tout ça dans un vrai Chromium : dépôt du fichier, progression, détections triées,
timecodes sur la grille de 3 s, curseur, lecture, arrêt automatique, changement de fichier en cours
d'analyse, plus zéro erreur console ou 404. Les comptages sont **exacts** (24 / 63 / 2 détections aux
seuils 0,25 / 0,05 / 0,70) et la première ligne est comparée à la vérité terrain : un curseur mort ou
une liste mélangée ne peuvent pas passer.

Il a déjà attrapé deux vraies casses : les fichiers `/ort/*.mjs` doivent être servis avec un **type
MIME JavaScript** sinon le backend WASM ne démarre pas — à vérifier sur votre hébergeur — et
`pnpm dev` était **totalement inutilisable** tant que ces fichiers vivaient dans `public/` (Vite
refuse de servir un fichier de `public/` atteint par un `import`, y compris l'import dynamique
qu'ORT fait de son runtime). Ils sont maintenant servis par un plugin Vite, en dev comme en build.

## Suite

P2 spectrogramme + timeline · P3 micro en direct · P4 filtre géo-temporel.

Le modèle géo de P4 est déjà récupéré et vérifié : entrée `[1, 3]` = latitude, longitude, semaine ;
sortie `[1, 6522]` probabilités de présence. Contrôle de bon sens à Paris (48,85 / 2,35), semaine 20 :

```
Merle noir 0.999 · Corneille noire 0.995 · Pigeon ramier 0.992
Pinson des arbres 0.973 · Hirondelle rustique 0.959 · Fauvette à tête noire 0.954
```

Il ne contient que des opérations élémentaires (pas de FFT) : sa conversion ne demandera aucune des
précautions décrites plus haut, et P4 n'aura aucune dépendance réseau à débloquer.

## Structure

```
src/lib/birdnet/     le pipeline (audio, fenêtrage, modèle, worker, sigmoïde, labels)
scripts/             récupération des artefacts, conversion ONNX, parité, bench
public/models/       poids + labels + LICENSE (générés, non versionnés)
```

## Licences

- **Code** de ce dépôt : MIT.
- **Modèle BirdNET** : CC BY-NC-SA 4.0 — voir `public/models/LICENSE`, généré avec les poids.
  Usage non commercial uniquement.

Citation :

> Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning solution for avian
> diversity monitoring. *Ecological Informatics*, 61, 101236.

## Résultats de parité actuels

Sur `soundscape.wav` (40 fenêtres × 6 522 classes = 260 880 comparaisons par niveau) :

| Niveau | `max|Δscore|` | Détections divergentes à 0,25 | Verdict |
|---|---|---|---|
| A — modèle seul | 6,8 × 10⁻⁵ | **0** | ✅ |
| A′ — 35 fenêtres zero-paddées | **1,7 × 10⁻²** | **0** | ⚠️ hors contrat, [documenté](docs/LIMITATIONS.md) |
| B — chaîne complète (Chromium) | 9,4 × 10⁻⁵ | **0** | ✅ |
| C — rééchantillonné 44,1 kHz | 1,4 × 10⁻³ | **0** | mesuré, non imposé |
| D — entrées dégénérées | 9,0 × 10⁻⁴ | **0** | ✅ |

Sur de l'audio réel, on est un ordre de grandeur sous le seuil demandé de 1 × 10⁻³ (niveaux A et B).
**Zéro détection divergente à tous les niveaux, y compris ceux qui ne sont pas imposés** — c'est le
résultat qui compte le plus, et il tient même là où l'écart de score sort du contrat.

Le niveau A′ est la limite connue : 14 des 35 fenêtres paddées dépassent 1 × 10⁻³, jusqu'à
1,7 × 10⁻². Une seule fenêtre par fichier est concernée, et aucune détection ne bouge. Cause et
bornes dans [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

Détail du niveau D :

| Entrée | `max|Δscore|` |
|---|---|
| silence total / continu à 0,5 | 5,1 × 10⁻⁴ |
| **moitié audio / moitié silence** | **9,0 × 10⁻⁴** |
| ton pur 4 kHz | 2,6 × 10⁻⁴ |
| créneau pleine échelle, bruit blanc, saturation, audio réel | ≤ 1,5 × 10⁻⁵ |

### Ce que le niveau C a mis au jour

Sur un fichier qui doit être rééchantillonné, `librosa.resample` (côté BirdNET) renvoie
**5 760 001** échantillons là où le ratio exact vaut 5 760 000,0 — un arrondi `ceil` sur un flottant.
Cet unique échantillon en trop crée une **41ᵉ fenêtre entière**, composée d'un échantillon réel et de
143 999 zéros.

Autrement dit : sur un fichier non-48 kHz, BirdNET-Analyzer et cet outil peuvent ne pas être d'accord
sur le *nombre de fenêtres analysées*. Le harnais le signale explicitement et compare les fenêtres
communes, plutôt que de masquer l'écart. L'écart de score qui subsiste (1,4 × 10⁻³) ne change **aucune**
détection au seuil de 0,25.

**Conseil pratique** : donner des fichiers déjà en 48 kHz. Ils ne sont alors ni rééchantillonnés ni
altérés, et le résultat est celui du niveau B.
