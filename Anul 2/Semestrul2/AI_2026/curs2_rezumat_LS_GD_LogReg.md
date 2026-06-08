# Rezumat Curs 2 — Least Squares, Gradient Descent & Regresie Logistică
**Surse:** `02_ML_LS_GD.pdf` + `2_1_Metoda_celor_mai_mici_patrate.md` + `2_2_Regresie_Liniara.md` + `2_3_Metoda_Gradient_Descent.md` + `2_4_Gradient_Descent.md` + `2_5_Stochastic_Gradient_Descent.md` + `2_6_Regresie_Logistica.md`
**UBB Informatică Anul 2 — Inteligență Artificială**

---

## Contextul cursului: Framework pentru regresii liniare

Cursul răspunde la 3 întrebări fundamentale:
1. **Ce modele de predicție sunt posibile?** → Ipotezele validate (modelul liniar)
2. **Cât de bun este un model?** → Funcția de cost (loss function)
3. **Cum se identifică cel mai bun predictor?** → Algoritmul de optimizare

Există două metode principale pentru găsirea celui mai bun model liniar:
- **Metoda celor mai mici pătrate** — soluție directă (analitică)
- **Gradient Descent** — soluție iterativă

---

## 1. Metoda Celor Mai Mici Pătrate (Least Squares)

### 1.1 Problema de regresie

**Date:**
- Vectori de intrare: $x_i \in \mathbb{R}^d$, $i = 1, \ldots, N$
- Valori de ieșire (etichete): $y_i \in \mathbb{R}$

**Scop:** Găsirea unui model liniar $f$ care transformă orice $x_i$ în $y_i$.

### 1.2 Modelul liniar

$$f(x) = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_d x_d$$

**Notație vectorială** — adăugăm 1 la vectorul de intrare pentru a include bias-ul:

$$\tilde{x} = (1, x_1, x_2, \ldots, x_d)^T \in \mathbb{R}^{d+1}$$

$$w = (w_0, w_1, \ldots, w_d)^T \in \mathbb{R}^{d+1}$$

$$f(x) = \tilde{x}^T w$$

> **Exemplu:** Prețul unui imobil cu $d = 2$ caracteristici (nr. camere $x_1$, suprafață $x_2$):
> $f(x) = w_0 + w_1 \cdot x_1 + w_2 \cdot x_2$
> Dacă $w_0 = 10$, $w_1 = 50$, $w_2 = 2$, atunci un imobil cu 3 camere și 60 m² are prețul prezis: $10 + 50 \cdot 3 + 2 \cdot 60 = 280$ (mii lei)

### 1.3 Funcția de cost (Loss function) — SSE

Suma pătratelor erorilor (Sum of Squared Errors):

$$\text{Loss}(w) = \sum_{i=1}^{N} (y_i - f(x_i))^2$$

**Forma matriceală:**

$$X = \begin{pmatrix} 1 & x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\ \vdots & & & & \vdots \\ 1 & x_{N,1} & x_{N,2} & \cdots & x_{N,d} \end{pmatrix} \in \mathbb{R}^{N \times (d+1)}$$

$$\text{Loss}(w) = \|y - Xw\|^2$$

### 1.4 Soluția optimă (Formula normală)

Minimizând funcția de cost prin derivare și egalare cu 0:

$$w = (X^T X)^{-1} X^T y$$

> Aceasta este formula de **învățare** — calculul direct al coeficienților optimi.

### 1.5 Cazul simplu: d = 1 (o singură variabilă)

$$f(x) = w_0 + w_1 x$$

$$w_1 = \frac{\text{cov}(x, y)}{\text{var}(x)}, \qquad w_0 = \bar{y} - w_1 \bar{x}$$

Unde $\bar{x}$, $\bar{y}$ sunt mediile variabilelor.

> **Exemplu numeric:** Avem 3 exemple de antrenament:
>
> | $x$ (nr. camere) | $y$ (preț) |
> |---|---|
> | 1 | 1 |
> | 2 | 3 |
> | 4 | 3 |
>
> $\bar{x} = \frac{1+2+4}{3} = \frac{7}{3}$, $\bar{y} = \frac{1+3+3}{3} = \frac{7}{3}$
>
> Se calculează $w_1$ și $w_0$ cu formulele de mai sus, obținând dreapta de regresie.

### 1.6 Eroarea medie pătratică (MSE)

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2 = \frac{\text{SSE}}{N}$$

> **Atenție la examen:** SSE și MSE diferă prin factorul $\frac{1}{N}$. Cursul oficial folosește SSE (fără împărțire la N). Fișierele .md folosesc uneori MSE. Ambele se minimizează la același $w$ optim.

### 1.7 Graficul reziduurilor (Residual Plot)

Reziduurile sunt $y_i - f(x_i)$ pentru fiecare punct.

- **Reziduuri aleatorii (fără pattern)** → model bun
- **Pattern curbat** → relație neliniară, modelul liniar nu e adecvat
- **Outlieri** → date care influențează puternic modelul
- **Heteroscedasticitate** → variabilitatea reziduurilor crește cu valorile prezise

### 1.8 Anscombe's Quartet

Patru seturi de date cu **aceiași coeficienți de regresie, medii și varianțe**, dar distribuții complet diferite vizual. Demonstrează că **vizualizarea datelor este esențială** — statistica descriptivă singură poate fi înșelătoare.

---

## 2. Gradient Descent (GD)

### 2.1 Motivație și intuiție

Metoda celor mai mici pătrate necesită calculul $(X^T X)^{-1}$, care poate fi costisitor sau imposibil pentru date mari. Gradient Descent oferă o alternativă **iterativă**.

**Intuiție:** Ești pe un deal, ai ochii închiși. La fiecare pas simți înclinația terenului și faci un pas mic în direcția coborârii. Repeți până ajungi în vale (minimul funcției de cost).

**Scop formal:**

$$\min_w \text{TrainLoss}(w)$$

**Definiție gradient:** direcția care modifică loss-ul **cel mai mult** (creștere maximă). Noi mergem în **direcția opusă** gradientului.

### 2.2 Algoritmul Gradient Descent

**Inițializare:** $w = [0, 0, \ldots, 0]$

**La fiecare iterație (epocă) $t$:**

$$w_k(t+1) = w_k(t) - \eta \cdot \text{error}(t) \cdot x_k, \quad k = 1, 2, \ldots, d$$

$$w_0(t+1) = w_0(t) - \eta \cdot \text{error}(t)$$

Unde:
- $\eta$ (eta) = **learning rate** (rata de învățare) — notat și $\alpha$ în unele surse
- $\text{error}(t) = f(x_t) - y_t = w_0(t) + w_1(t) x_1 + \ldots + w_d(t) x_d - y$ (predicție − realitate)

**Forma compactă (cu gradient):**

$$w \leftarrow w - \eta \cdot \nabla_w \text{TrainLoss}(w)$$

**Calculul gradientului** (prin regula lanțului):

$$\text{TrainLoss}(w) = \frac{1}{|\mathcal{D}_\text{train}|} \sum_{(x,y) \in \mathcal{D}_\text{train}} (w \cdot \phi(x) - y)^2$$

$$\nabla_w \text{TrainLoss}(w) = \frac{1}{|\mathcal{D}_\text{train}|} \sum_{(x,y) \in \mathcal{D}_\text{train}} 2(\underbrace{w \cdot \phi(x) - y}_{\text{predicție} - \text{target}}) \phi(x)$$

### 2.3 Rolul ratei de învățare $\eta$

| $\eta$ | Efect |
|--------|-------|
| Prea mic | Convergență lentă, multe iterații necesare |
| Optim | Convergență rapidă și stabilă |
| Prea mare | Sare peste minim, poate diverge |

> **Exemplu:** Dacă $\eta = 0.0001$ și minimul e la $w=3$, algoritmul face pași minusculi și are nevoie de mii de iterații. Dacă $\eta = 10$, sare de la $w=0$ la $w=20$, apoi la $w=-14$, oscilând haotic.

### 2.4 Criteriile de oprire

- Când gradientul e aproape zero: $\nabla_w \text{TrainLoss}(w) \approx 0$
- Când **step size** $= \text{Slope} \times \eta \leq 0.001$ (sau alt prag mic)
- Când s-a atins numărul maxim de pași (ex: 1000 iterații)

---

## 3. Versiunile Gradient Descent

### 3.1 Batch Gradient Descent (GD clasic)

- **Eroarea** se calculează pe **toate exemplele** de antrenament
- **Update** se face **la finalul unei epoci** (după ce toate exemplele au fost evaluate)
- **Formula:**
$$w \leftarrow w - \eta \cdot \nabla_w \text{TrainLoss}(w)$$

| Avantaje | Dezavantaje |
|----------|-------------|
| Stabil, direcție precisă | Lent pentru seturi mari |
| Convergență netedă | Necesită toată memoria pentru un update |

### 3.2 Stochastic Gradient Descent (SGD)

- **Eroarea** se calculează pentru **fiecare exemplu** individual
- **Update** se face **după fiecare exemplu** (online learning)
- **Formula** (pentru exemplul $(x^{(i)}, y^{(i)})$):

$$\hat{y}^{(i)} = w^T x^{(i)} \quad \text{(predicție)}$$

$$e^{(i)} = \hat{y}^{(i)} - y^{(i)} \quad \text{(eroarea)}$$

$$w_j := w_j - \alpha \cdot e^{(i)} \cdot x_j^{(i)}, \quad j = 1, \ldots, d$$

$$w_0 := w_0 - \alpha \cdot e^{(i)}$$

> **Termenul „stochastic"** vine de la faptul că direcția de update conține zgomot (folosim un singur exemplu, nu toată informația), dar în medie se apropie de minimum.

| Avantaje | Dezavantaje |
|----------|-------------|
| Rapid | Oscilant, zgomotos |
| Bun pentru date mari | Poate să nu convergă exact |
| Poate evita minime locale | Sensibil la $\alpha$ |

**Tehnici de îmbunătățire SGD:**

**Learning rate decay:**
$$\alpha_t = \frac{\alpha_0}{1 + k \cdot t}$$

**Momentum:**
$$v_j := \gamma v_j + \alpha \cdot \frac{\partial J^{(i)}(w)}{\partial w_j}, \qquad w_j := w_j - v_j$$
Unde $\gamma \approx 0.9$ adaugă „inerție" pentru a trece peste văi înguste.

### 3.3 Mini-batch Gradient Descent

- Setul de date se împarte în **mini-batch-uri** (grupuri mici)
- **Eroarea** se calculează pentru exemplele dintr-un mini-batch
- **Update** se face după fiecare mini-batch
- **Compromis** între Batch GD și SGD

**Tabel comparativ complet:**

| Tip | Date per update | Când se face update | Viteză | Stabilitate | Utilizare tipică |
|-----|----------------|---------------------|--------|-------------|-----------------|
| Batch GD | Toate (N) | La finalul epocii | Lent | Foarte stabil | Date mici |
| SGD | 1 exemplu | Dopo fiecare exemplu | Rapid | Oscilant | Date foarte mari |
| Mini-batch | Subset mic (ex: 32) | După fiecare batch | Echilibru | Echilibru | Deep learning |

> **Epocă** = o trecere completă prin toate datele de antrenament.

---

## 4. Regresie Logistică (Clasificare)

### 4.1 Motivație — De ce nu regresie liniară?

Dacă avem o problemă de **clasificare binară** (ieșire 0 sau 1) și folosim regresie liniară, modelul poate produce valori ca 1.8 sau -0.3, care nu au sens ca probabilități. Regresia logistică corectează asta.

### 4.2 Structura problemei

**Date:**
- Intrări: $x_i \in \mathbb{R}^d$
- Ieșiri: $y_i \in \{0, 1\}$ (clasificare binară)

**Tipuri de clasificare:**
- **Binar:** Pass/Fail, Spam/Non-spam, Bolnav/Sănătos
- **Multi-clasă:** Cat/Dog/Panda
- **Ordinal:** Low/Medium/High

### 4.3 Modelul

**Pasul 1 — funcție liniară** (ca la regresia liniară):

$$z = f(x) = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_d x_d$$

**Pasul 2 — funcția Sigmoid** (transformă $z$ în probabilitate):

$$S(z) = \frac{1}{1 + e^{-z}}$$

**Proprietăți sigmoid:**
- $z \to +\infty \Rightarrow S(z) \to 1$
- $z \to -\infty \Rightarrow S(z) \to 0$
- $z = 0 \Rightarrow S(z) = 0.5$
- Mapează orice număr real în intervalul $(0, 1)$

**Ieșirea finală (probabilitate):**

$$\hat{y} = S(f(x)) = P(y = 1 \mid x) = \frac{1}{1 + e^{-f(x)}}$$

### 4.4 Pragul de decizie (threshold)

$$\hat{y} \geq 0.5 \Rightarrow \text{clasa 1} \qquad \hat{y} < 0.5 \Rightarrow \text{clasa 0}$$

> Pragul 0.5 este standard, dar poate fi ajustat. Ex: în medicină, poate fi setat la 0.3 — mai bine să clasifici greșit un om sănătos ca bolnav decât invers.

### 4.5 Antrenarea — Gradient Descent pentru regresie logistică

**Inițializare:** $w_0, w_1, \ldots, w_d$ (aleator sau 0)

**La fiecare iterație $t$ (pentru exemplul curent):**

$$\text{error}(t) = S(w_0(t) + w_1(t) x_1 + \ldots + w_d(t) x_d) - y$$

$$w_k(t+1) = w_k(t) - \eta \cdot \text{error}(t) \cdot x_k, \quad k = 1, \ldots, d$$

$$w_0(t+1) = w_0(t) - \eta \cdot \text{error}(t)$$

> **Diferența față de regresia liniară:** eroarea se calculează pe **Sigmoid(predicție) − realitate**, nu pe predicție − realitate direct.

### 4.6 Funcția de cost: Binary Cross Entropy

$$J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 4.7 Exemplu practic — Predicția promovării unui examen

| Student | $x_1$ (ore studiu) | $x_2$ (ore somn) | $y$ (promovat) |
|---------|-------------------|-----------------|----------------|
| A | 5 | 6 | 1 |
| B | 2 | 4 | 0 |

Modelul: $f(x) = w_0 + w_1 x_1 + w_2 x_2$

1. Calculezi $f(x)$ pentru exemplul curent
2. Aplici sigmoid: $\hat{y} = S(f(x))$
3. Dacă $\hat{y} \geq 0.5$ → prezici 1 (promovat); altfel → 0
4. Calculezi eroarea: $e = \hat{y} - y$
5. Actualizezi coeficienții

---

## 5. Recapitulare Comparativă — Cele 3 Metode

| Caracteristică | Least Squares | Gradient Descent | Regresie Logistică |
|---|---|---|---|
| Tip problemă | Regresie | Regresie | **Clasificare** |
| Tip algoritm | Supervizat | Supervizat | Supervizat |
| Tip output | Continuu ($\mathbb{R}$) | Continuu ($\mathbb{R}$) | Discret (clase) |
| Metoda | Soluție directă | Iterativă | Iterativă (GD) |
| Formula cheie | $w = (X^T X)^{-1} X^T y$ | $w \leftarrow w - \eta \nabla_w \text{Loss}$ | + Sigmoid pe output |
| Funcție de cost | SSE / MSE | SSE / MSE | Binary Cross Entropy |
| Când se folosește | Date mici, soluție exactă | Date mari, optimizare | Clasificare binară/multi |

---

## 6. ⚠️ Aspecte Critice pentru Examen

### Tipare din SUBIECT_EXAMEN_2025

**Subiect 2a** — *"Se dă un model liniar din grafic. Precizați prețul previzionat pentru 2 camere."*
→ Citești $w_0$ (intersecția cu axa Y) și $w_1$ (panta) din grafic, calculezi $f(2) = w_0 + w_1 \cdot 2$

**Subiect 2b** — *"Două modele cu date diferite. Care are eroarea medie pătratică mai mare?"*
→ Compari vizual împrăștierea punctelor față de dreaptă — modelul cu puncte mai departe de dreaptă are MSE mai mare

**Subiect 2c** — *"Relația nu e liniară. Cum construiești totuși un model liniar?"*
→ Aplici o transformare neliniară asupra datelor ($\log$, $\sqrt{}$, $x^2$) ca să liniarizezi relația, apoi aplici regresia liniară pe datele transformate

### Concepte-cheie de memorat

1. **Formula normală:** $w = (X^T X)^{-1} X^T y$ — soluție directă, fără iterații
2. **Formula GD:** $w_k := w_k - \eta \cdot \text{error} \cdot x_k$ — actualizare iterativă
3. **Eroarea în GD:** $\text{error} = \text{predicție} - \text{realitate}$ (cu semn!)
4. **Sigmoid:** $S(z) = \frac{1}{1+e^{-z}}$, mapează $\mathbb{R} \to (0,1)$
5. **Diferența GD vs SGD:** update după toate exemplele vs. după fiecare exemplu
6. **De ce sigmoid la regresie logistică:** pentru că ieșirea trebuie să fie o probabilitate între 0 și 1
7. **MSE vs SSE:** MSE = SSE/N — ambele au același minim

---

*Rezumat generat exclusiv din `02_ML_LS_GD.pdf` și fișierele .md corespunzătoare Cursului 2*
