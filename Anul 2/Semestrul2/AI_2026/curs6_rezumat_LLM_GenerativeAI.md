# Rezumat Curs 6 — AI Generativă & Large Language Models (LLMs)
**Surse:** `06_GenerativeAI_LLM_2026.pdf` + `5_1_LLMs___Large_Language_Models.md` + `5_3_Transformers.md` + `5_4_Masca_de_atentie_in_Transformer.md`
**UBB Informatică Anul 2 — Inteligență Artificială**

---

## 1. Ce este un LLM?

Un **Large Language Model (LLM)** este un model de rețea neuronală profundă antrenat pe cantități foarte mari de text, capabil să **înțeleagă și să genereze limbaj natural**.

### 1.1 Ingredientele unui LLM

Pentru a construi un LLM sunt necesare 3 componente:
1. **Model/algoritm inteligent** — arhitectura Transformer
2. **Date** — cantități enorme de text (Wikipedia, cărți, cod, conversații)
3. **Putere de calcul** — GPU/TPU

### 1.2 Proprietatea cheie: Autoregresivitate

LLM-urile sunt modele **autoregresive** — prezic **următorul token** pe baza tuturor tokenilor anteriori:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

> **Exemplu:** Propoziția "Broasca a mâncat..." → modelul prezice că urmează "musca" sau "o insectă", nu "o mașină".

### 1.3 GPT vs ChatGPT — Diferența importantă pentru examen

| | GPT-3 | ChatGPT |
|---|---|---|
| Ce este | Un **LLM** | Un **sistem** care conține un LLM |
| Funcție | Model de limbaj brut | Asistent conversațional |
| Antrenare suplimentară | Nu | Da (RLHF + Fine-tuning) |

> **Răspuns la Subiectul 5 din examen:** "ChatGPT este un sistem care conține un LLM" (**varianta d**).

---

## 2. Tokenizare

Înainte de procesare, textul se împarte în **tokeni** (unități elementare).

### 2.1 Tipuri de tokenizare

| Tip | Descriere | Exemplu | Folosit în |
|---|---|---|---|
| White Space | Separă pe spații | "eu merg" → ["eu", "merg"] | Simple |
| Word | Separă în cuvinte | "running" → ["running"] | Comun |
| Character | Separă în caractere | "chat" → ["c","h","a","t"] | CharFormer |
| **Subword (BPE)** | Împarte în sub-cuvinte comune | "running" → ["run","ning"] | **GPT, GPT-2, GPT-3** |
| WordPiece | Similar BPE, metoda Google | | **BERT** |
| Unigram | Probabilistic | | T5, ALBERT |

> **Byte Pair Encoding (BPE, 2016):** Cel mai frecvent în GPT. Pornește de la caractere și fuzionează progresiv perechile cele mai frecvente. Echilibru bun între vocabular mic și reprezentare eficientă.

### 2.2 Token-free models

Modele care nu folosesc tokeni: CharFormer, ByT5, MegaByte (2023), Byte-latent transformer (2024).

---

## 3. Word Embeddings

### 3.1 De ce avem nevoie de embedding-uri?

Modelele ML lucrează cu numere. Un cuvânt trebuie transformat într-un vector numeric care să păstreze **relații semantice**.

> **Exemplu de relație semantică:**
> $\text{Emb(king)} - \text{Emb(queen)} \approx \text{Emb(man)} - \text{Emb(woman)}$

### 3.2 Vectori rari vs. vectori densi

| | Vectori rari (one-hot, co-ocurență) | Vectori densi (embedding) |
|---|---|---|
| Lungime | Foarte mare ($\|V\| = 20.000-50.000$) | Scurtă (200-1000) |
| Elemente | Multe zerouri | Puține zerouri |
| Semantică | **Nu** păstrează | **Păstrează** |
| Generalizare | Slabă | Bună |

### 3.3 Tipuri de embedding-uri

#### Frequency-based (bazate pe frecvență)
- **One-hot encoding** → vector cu un singur 1, restul 0
- **TF-IDF** → ponderi bazate pe frecvența în document vs. corpus
- **Co-occurrence matrix** → câte ori apar cuvintele împreună

#### Prediction-based (learned embeddings — învățate)

**Statice (context-free):**
- **Word2Vec (2013)** — două variante:
  - *Skip-gram:* din cuvântul curent prezice contextul
  - *CBOW (Continuous Bag of Words):* din context prezice cuvântul curent
- **GloVe (2014)** — bazat pe statistici globale de co-ocurență
- **FastText (2016)** — lucrează cu sub-cuvinte (n-grams)

**Dinamice (context-based):**
- **ELMo (2018)** — primul model cu embeddings contextuale
- **BERT (2019)** — bidirectional, embeddings diferite pentru același cuvânt în contexte diferite
- **GPT** — unidirectional, autoregresiv

> **Diferența esențială:**
> - **Static:** "bank" are același embedding indiferent de context
> - **Dinamic:** "bank" (malul râului) ≠ "bank" (instituție financiară)

### 3.4 Arhitectura CBOW

**Scop:** Din contextul [she, ride] → prezice "bike"

Pași:
1. Fiecare cuvânt de context → vector one-hot
2. $W^T \times x_{she} = v_{she}$ (embedding)
3. $\hat{v} = \frac{v_{she} + v_{ride}}{2}$ (medie)
4. ${W'}^T \times \hat{v} = z$
5. $\hat{y} = \text{softmax}(z)$ → probabilitate pentru fiecare cuvânt din vocabular
6. Maximizăm probabilitatea cuvântului corect ("bike")

**Ce se învață:** matricele $W$ și $W'$ (ponderile rețelei) devin embedding-urile.

---

## 4. Antrenarea unui LLM

### 4.1 Pasul 1 — Pre-training (antrenare inițială)

**Scop:** Înțelegerea structurii limbajului natural.

**Sarcina:** Next-token prediction — dat un șir de cuvinte, prezice următorul.

**Funcția de cost minimizată:**

$$L = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$$

Unde:
- $x_t$ = tokenul de la poziția $t$ (ce trebuie prezis)
- $x_1, \ldots, x_{t-1}$ = toți tokenii anteriori (contextul)
- $P$ = probabilitatea atribuită de model tokenului corect

> Cu cât modelul e mai sigur pe răspunsul corect, cu atât $\log P$ e mai aproape de 0, deci $L$ e mai mic.

**Date:** Wikipedia, cărți, cod GitHub, conversații, documentații tehnice — **nesupervizat** (nu necesită etichete manuale).

### 4.2 Pasul 2 — Fine-tuning (aliniere și specializare)

**Scop:** Dobândirea de competențe specifice (ex: răspuns politicos, cod, rezumare).

**Metoda:** Date etichetate — perechi (Prompt $X$, Răspuns așteptat $Y$) — **supervizat**.

> Exemplu: GPT-3 (pre-trained) → + fine-tuning pe conversații → ChatGPT

### 4.3 Pasul 3 — RLHF (Reinforcement Learning from Human Feedback)

**Scop:** Îmbunătățirea calității răspunsurilor conform preferințelor umane.

**Pași:**
1. Modelul generează **mai multe răspunsuri** pentru același prompt
2. Un om le **ordonează** de la cel mai bun la cel mai slab
3. Se antrenează un **reward model** care învață să prezică preferințele umane
4. Modelul LLM se antrenează cu **PPO** (algoritm RL) pentru a maximiza recompensa

**Rezultat:** Răspunsuri mai utile, mai sigure, mai naturale.

---

## 5. Arhitectura Transformer

Introdusă în 2017 în lucrarea **"Attention is All You Need"** (Vaswani et al.). A înlocuit complet RNN-urile și LSTM-urile pentru procesarea textului.

### 5.1 Avantajul față de RNN/LSTM

| | RNN / LSTM | Transformer |
|---|---|---|
| Procesare | **Secvențială** (token cu token) | **Paralelă** (toți tokenii simultan) |
| Context lung | Dificil (vanishing gradient) | Ușor (atenție directă) |
| Viteză antrenare | Lentă | **Rapidă** |

### 5.2 Componentele unui Transformer

#### 1. Input Embeddings + Positional Encoding

Fiecare token → vector embedding. Dar Transformer-ul nu știe ordinea cuvintelor (procesează în paralel), deci se adaugă **Positional Encoding**:

$$\text{Embedding final} = \text{WordEmbedding} + \text{PositionalEncoding}$$

Formule PE:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)$$

Unde:
- $pos$ = poziția tokenului în secvență
- $i$ = dimensiunea curentă a embedding-ului
- $d$ = dimensiunea totală a embedding-ului

#### 2. Self-Attention

**Intuiție:** Mecanismul care decide cât de mult "se uită" fiecare token la celelalte tokene.

**Analogia cu baza de date:**
- **Query (Q)** = "ce caut?" — ce întreabă tokenul curent
- **Key (K)** = "ce ofer?" — ce descrie fiecare token
- **Value (V)** = "ce conțin?" — informația efectivă a fiecărui token

**Formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Unde:
- $Q \in \mathbb{R}^{v \times d}$ = matricea de query-uri
- $K \in \mathbb{R}^{v \times d}$ = matricea de key-uri
- $V \in \mathbb{R}^{v \times n}$ = matricea de value-uri
- $d_k$ = dimensiunea vectorului key (factor de scalare)
- $v$ = numărul de cuvinte/tokeni
- $n$ = lungimea embedding-ului
- $d$ = numărul de features abstracte

**De ce împărțim la $\sqrt{d_k}$?** — Produsele scalare pot deveni foarte mari pentru $d_k$ mare, dând softmax valori prea extreme (gradienți mici). Împărțirea stabilizează antrenarea.

**Pașii calculului:**

| Pas | Operație | Dimensiuni |
|---|---|---|
| 1. Transformare | $input \times K$, $Q^T \times input^T$ | $(v,n) \to (v,d)$ și $(d,v)$ |
| 2. Scoruri | $(v,d) \times (d,v) = (v,v)$ | Matrice de similarități |
| 3. Scalare | $(v,v) / \sqrt{d_k}$ | Stabilizare |
| 4. Normalizare | $\text{softmax}(v,v)$ | Suma pe linie = 1 |
| 5. Reponderare | $(v,v) \times (v,n) = (v,n)$ | Tokeni reponderați |

> **Exemplu:** Propoziția "Broasca a mâncat musca."
> Când procesăm tokenul "mâncat", atenția față de "broasca" (subiectul) va fi mare, față de "musca" (obiectul) va fi medie, față de "a" va fi mică.

#### 3. Multi-Head Attention

În loc de un singur mecanism de atenție, se folosesc **$h$ capete în paralel**, fiecare învățând relații diferite:

$$Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V$$

$$head_i = \text{Attention}(Q_i, K_i, V_i)$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \ldots, head_h) \cdot W^O$$

> Un cap poate învăța relații sintactice (subiect-verb), altul relații semantice (sinonime), altul relații de co-referință (pronume-substantiv).

#### 4. Feed-Forward Network (FFN)

Aplicat fiecărui token independent, după atenție:

$$FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

Transformă fiecare token într-o reprezentare mai abstractă, fără a combina tokeni între ei.

#### 5. Normalization + Residual connections

Fiecare sub-strat (atenție sau FFN) are:
- **Add:** conexiune rezidual ($x + \text{sublayer}(x)$) — previne vanishing gradient
- **Norm:** Layer Normalization — stabilizează antrenarea

#### 6. Softmax final

Stratul de ieșire produce probabilități pentru fiecare token din vocabular → se alege tokenul cu probabilitatea maximă.

### 5.3 Tipuri de arhitecturi Transformer

| Tip | Componente | Sarcini | Exemple |
|---|---|---|---|
| **Encoder only** | Stiva de encodere | Clasificare, analiză sentiment | BERT, RoBERTa |
| **Decoder only** | Stiva de decodere | Generare text, completare | **GPT, GPT-2, GPT-3** |
| **Encoder-Decoder** | Ambele | Traducere, rezumare | BART, T5 |

### 5.4 Cât de mare e un LLM? — Parametrii GPT-2

Notații:
- $V$ = dimensiunea vocabularului
- $E$ = dimensiunea embedding-ului (model dimension)
- $L$ = lungimea secvenței de intrare (context window)
- $N_B$ = numărul de blocuri Transformer
- $H$ = dimensiunea stratului ascuns din MLP

**Formula totală parametri:**

$$\#param = E \times (V + L) + N_B \times (4E^2 + 9E + 2EH + H) + 2E + E \times V$$

**Exemplu GPT-2:**

| $V$ | $E$ | $L$ | $N_B$ | $H$ | Total parametri |
|---|---|---|---|---|---|
| 50.257 | 768 | 1.024 | 12 | 3.072 | **~124 milioane** |

**GPT-3:** 175 miliarde parametri. **GPT-4:** ~1.76 trilioane (estimat, MoE).

---

## 6. Arhitecturi specifice

### 6.1 BERT

- **Bidirectional Encoder Representations from Transformers**
- Procesează textul în **ambele direcții** simultan
- Pre-training prin **Masked Language Modeling** (prezice cuvinte mascate din mijlocul propoziției)
- Versiuni: Base (12 layere, 768 hidden, 12 attention heads), Large (24 layere, 1024 hidden, 16 heads)
- Utilizare: Clasificare, analiză sentiment, NER

### 6.2 GPT (Generative Pre-trained Transformer)

- **Unidirectional Decoder** — procesează stânga → dreapta
- Pre-training prin **Next-token prediction**
- Generează text token cu token (autoregresiv)
- GPT-3: 175B parametri, few-shot learning fără fine-tuning

---

## 7. Prompt Engineering

**Definiție:** Arta de a formula inputul (promptul) pentru a obține rezultatul dorit de la un LLM.

### Tipuri de prompting

| Tip | Descriere | Exemplu |
|---|---|---|
| **Zero-shot** | Fără exemple | "Traduce în română: Hello" |
| **One-shot** | Cu un singur exemplu | "EN: Hello → RO: Bună. EN: Bye → RO: ?" |
| **Few-shot** | Cu câteva exemple | Multiple perechi EN→RO |
| **Chain of Thought** | Cere pași intermediari | "Gândește pas cu pas..." |
| **Zero-shot CoT** | "Let's think step by step" | Fără exemple, dar cu instrucțiune de raționament |

---

## 8. Limitări și Pericole

| Problemă | Descriere |
|---|---|
| **Halucinații** | Modelul "inventează" fapte false cu încredere |
| **Bias** | Reflectă prejudecăți din datele de antrenament |
| **Dezinformare** | Poate genera texte convingătoare dar incorecte |
| **Lipsa memoriei** | Nu păstrează context pe termen lung fără arhitecturi speciale |
| **Nedeterminist** | Același prompt → răspunsuri diferite |

---

## 9. Direcții viitoare

- **LLM-uri multimodale** — text + imagine + video + audio (ex: GPT-4o)
- **Memorie extinsă** — context windows din ce în ce mai mari (2025: ~2M tokens; 2026: ~10M tokens)
- **Agenți autonomi** — modele care planifică și acționează
- **Raționament simbolic** — integrarea gândirii structurate

---

## 10. ⚠️ Aspecte Critice pentru Examen

### Tipare din SUBIECT_EXAMEN_2025

**Subiect 5** — *"Care afirmații sunt corecte și complete?"*
- a. GPT-3 este un LLM ✓
- b. ChatGPT este un LLM ✗ (e un sistem care conține un LLM)
- c. GPT-3 este un sistem care conține un LLM ✗
- **d. ChatGPT este un sistem care conține un LLM ✓** ← Răspuns corect

**Subiect 18** — *"Care afirmație e adevărată pentru embedding-urile prin one-hot encoding?"*
- **a. Vectorii conțin valori de 0 și 1** ← Răspuns corect
- b. Surprind relații semantice ✗
- c. Ineficienți pentru vocabulare lungi (invers — sunt ineficienți pentru orice vocabular mare)
- d. Conțin doar valori numerice ✗ (prea general)

### Concepte cheie de memorat

1. **LLM** = rețea neuronală adâncă antrenată pe text masiv pentru a prezice tokenul următor
2. **GPT-3 = LLM; ChatGPT = sistem care conține un LLM**
3. **Autoregresiv** = prezice tokenul următor pe baza celor anteriori
4. **BPE** = tokenizarea folosită de GPT (Byte Pair Encoding)
5. **One-hot** → valori 0/1, nu păstrează semantica; **Embedding** → valori reale, păstrează semantica
6. **CBOW** = din context prezice cuvântul țintă
7. **Pre-training** = next-token prediction, nesupervizat, date masive
8. **Fine-tuning** = specializare pe sarcini specifice, supervizat
9. **RLHF** = ajustare după preferințe umane (3 pași: generare → ordonare umană → reward model → PPO)
10. **Attention formula:** $\text{softmax}(QK^T/\sqrt{d_k}) \cdot V$
11. **Q = query, K = key, V = value** — analogia cu baza de date
12. **$\sqrt{d_k}$** = factor de scalare pentru stabilizare
13. **Multi-head attention** = mai multe capete în paralel, fiecare învață relații diferite
14. **Encoder-only** = BERT (clasificare); **Decoder-only** = GPT (generare)
15. **Halucinații** = LLM-ul inventează fapte false cu aparentă încredere

---

*Rezumat generat exclusiv din `06_GenerativeAI_LLM_2026.pdf` și fișierele .md 5_1, 5_3, 5_4*
