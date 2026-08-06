# Limites connues

Ce fichier documente les écarts mesurés avec BirdNET officiel. Ils sont **mesurés et bornés**, pas
compensés : aucun facteur de correction n'est appliqué nulle part dans le pipeline.

---

## 1. Fenêtre finale zero-paddée : jusqu'à 1,7 × 10⁻² d'écart de score

**C'est la limite la plus importante de P0.** Elle sort du contrat de 1 × 10⁻³.

### Ce qui est mesuré

`pnpm parity` (niveau A′) balaie 35 fenêtres paddées — 5 positions dans le fichier × 7 longueurs de
queue — en comparant notre ONNX au tflite FP32 officiel :

| Audio réel conservé dans la dernière fenêtre | pire `max\|Δscore\|` | vs contrat 1e-3 |
|---|---|---|
| 0,10 s | 3,5 × 10⁻⁴ | ok |
| 0,25 s | 6,9 × 10⁻⁴ | ok |
| **0,50 s** | **1,3 × 10⁻²** | **dépassé** |
| **0,75 s** | **1,7 × 10⁻²** | **dépassé (pire cas)** |
| **1,00 s** | **1,0 × 10⁻²** | **dépassé** |
| **1,50 s** | **1,4 × 10⁻³** | **dépassé** |
| 2,50 s | 5,8 × 10⁻⁴ | ok |

14 des 35 fenêtres dépassent le contrat. **Détections franchissant le seuil de 0,25 différemment :
0 sur 35.**

### Portée réelle

Une seule fenêtre par fichier est concernée : la dernière, et seulement si la durée du fichier n'est
pas un multiple de 3 s. Toutes les autres fenêtres restent à ≤ 9,4 × 10⁻⁵ (niveau B, chaîne complète).

Ce n'est **pas** une différence de comportement avec BirdNET : le paquet `birdnet` 0.2.16 fait
exactement la même chose (« *fill last segment with silence up to segmentsize if it is smaller than
3s* », `acoustic/inference/core/producer.py`). Le fenêtrage est identique ; seule l'arithmétique
diffère.

### Cause, précisément

Le front-end mel a été replié d'une FFT vers un produit scalaire direct de 2048 termes (voir README).
Les deux calculent le même opérateur linéaire, mais pas avec la même stabilité numérique :

- une FFT accumule en `log₂(2048) = 11` étages → erreur d'arrondi en `O(√log N)` ;
- un produit scalaire direct accumule 2048 termes → erreur en `O(√N)`.

Sur une trame quasi silencieuse, `Re(DFT)[k]` vaut ~0 pour tout `k ≥ 2` : le résultat *est* le
résidu d'arrondi. Mesuré sur MEL_SPEC1 contre une référence float64, sur une trame constante :

| | erreur vs float64 |
|---|---|
| convolution repliée (nous) | 1,16 × 10⁻² |
| STFT (tflite officiel) | 2,51 × 10⁻³ |
| magnitude médiane de la feature | 1,44 × 10⁻³ |

Le repli est ~4,6× moins bien conditionné que la FFT, et **les deux** ont une erreur supérieure à la
grandeur qu'elles calculent. Autrement dit, dans ce régime la référence elle-même n'a plus de
signification numérique à cette échelle — mais c'est bien notre écart *par rapport à elle* qui définit
le contrat, donc le dépassement est réel et compte.

Le `x^(1/(1+e^s))` du modèle (exposant ≈ 0,226, dérivée infinie en 0) amplifie ensuite ce résidu.

### Pourquoi ce n'est pas corrigé

Corriger demanderait de retrouver la stabilité de la FFT, c'est-à-dire de factoriser la matrice repliée
en étages — soit réimplémenter une FFT, ce que la spec du projet exclut explicitement, et que le
backend WASM d'ONNX Runtime Web ne permet pas d'éviter autrement (pas d'opérateur `DFT` fiable).

Options si cette limite devient gênante :

1. **Ne rien faire** — aucune détection n'a changé sur 35 fenêtres testées, et une fenêtre à 75 % de
   silence ne porte de toute façon pas d'information exploitable.
2. **Ignorer ou marquer** les fenêtres composées à plus de X % de padding.
3. **Rogner** le fichier à un multiple de 3 s (supprime le cas entièrement).

Le harnais impose une borne de régression de 3 × 10⁻² (≈ 2× le pire mesuré) et **exige zéro
divergence de détection**. Il ne masque rien : le tableau ci-dessus est réimprimé à chaque exécution.

---

## 2. Rééchantillonnage : 1,4 × 10⁻³, et un décalage d'une fenêtre

Voir README, niveau C. Deux effets sur un fichier qui n'est pas en 48 kHz :

- **Écart de score de 1,4 × 10⁻³** entre le rééchantillonneur du navigateur et `resampy`. Aucune
  détection ne change.
- **`librosa.resample` renvoie 5 760 001 échantillons** là où le ratio exact vaut 5 760 000,0 (arrondi
  `ceil` sur un flottant). Cet échantillon en trop crée une **41ᵉ fenêtre** entière contenant un
  échantillon réel et 143 999 zéros. BirdNET-Analyzer et cet outil peuvent donc analyser un nombre de
  fenêtres différent sur le même fichier.

**Conseil** : fournir des fichiers déjà en 48 kHz. Ils ne sont alors ni rééchantillonnés ni altérés,
et les deux effets disparaissent.

---

## 3. `resampleTo48k` : chemin de repli non couvert et de qualité inférieure

Sous Chromium et Firefox, `decodeAudioData` rééchantillonne vers la fréquence du contexte de décodage,
donc `resampleTo48k` **n'est jamais atteint** — le niveau C mesure en réalité le rééchantillonneur du
*décodeur*, pas celui-là.

La fonction reste nécessaire pour les moteurs qui renvoient la fréquence propre du fichier (Safari
historiquement), mais mesurée sous Chromium elle est de qualité inférieure :

- **aucun filtre anti-repliement au sous-échantillonnage** : un ton à 30 kHz dans un fichier 96 kHz
  ressort à 18 kHz à gain unitaire. Pour de la bioacoustique avec contenu ultrasonore (chiroptères,
  orthoptères), c'est un repliement direct dans la bande des oiseaux ;
- **interpolation d'ordre faible** sur les ratios non entiers : raies parasites à −27,6 dB pour une
  entrée à 15 kHz, contre < −80 dB pour `resampy kaiser_fast`.

Non corrigé en P0, et non couvert par le harnais. À traiter si Safari devient une cible, ou en
pré-convertissant en 48 kHz.

---

## 4. Couverture du harnais

- `TOP_K_PER_WINDOW = 64` n'est jamais atteint en pratique. Mesuré sur la fixture avec le plancher
  d'analyse réel de l'interface (0,01, hors classes non-oiseaux) : **27 classes au maximum** dans la
  fenêtre la plus chargée, pour 420 détections au total. Sur des entrées pathologiques (bruit blanc
  11, bruit rose 0, continu/saturé 0, silence 0) on reste très en dessous. La troncature est donc un
  filet de sécurité, et depuis P1 elle est **signalée** (`truncatedWindows` / `WindowResult.truncated`)
  au lieu de raccourcir la liste en silence.
- Les mesures de performance sont Linux x86_64 / Chromium. Les chiffres Mac Apple Silicon et Safari
  demandés n'ont pas pu être produits ici (voir `PERF.md`).
