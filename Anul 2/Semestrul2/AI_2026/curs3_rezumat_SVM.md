# Rezumat Curs 3 — Mașini cu Suport Vectorial (MSV / SVM)
**Surse:** `03_ML_SVM.pdf` + `3_1_Masini_cu_suport_vectorial__MSV_.md`
**UBB Informatică Anul 2 — Inteligență Artificială**

---

## 1. Ce sunt Mașinile cu Suport Vectorial?

### 1.1 Definiție

Mașinile cu suport vectorial (MSV, sau SVM în engleză — Support Vector Machines) sunt **algoritmi de învățare supervizată** folosiți pentru **clasificare**. Ideea lor de bază: găsirea unui **hiperplan** care separă clasele cu o **margine maximă**.

Au fost dezvoltate de Vapnik în anii 1970 și popularizate după 1992.

### 1.2 Ce este un hiperplan?

Un hiperplan este un concept geometric care depinde de dimensiunea datelor:

| Nr. dimensiuni ($m$) | Forma separatorului | Ecuația |
|---|---|---|
| $m = 1$ | Un punct (prag) | $w_1 x + b = 0$ |
| $m = 2$ | O dreaptă | $w_1 x_1 + w_2 x_2 + b = 0$ |
| $m = 3$ | Un plan | $w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$ |
| $m > 3$ | Un **hiperplan** | $\sum_{i=1}^{m} w_i x_i + b = 0$ |

> **Exemplu 2D:** Imaginează-ți puncte portocalii (clasa 1) și verzi (clasa 2) pe o foaie. SVM-ul trasează **dreapta** care le separă cel mai bine.

### 1.3 Notații

| Simbol | Ce înseamnă | Exemplu |
|---|---|---|
| $x^d \in \mathbb{R}^m$ | Vectorul de atribute al exemplului $d$ | $x = (vârstă, greutate)$ → $m = 2$ |
| $t^d \in \{-1, 1\}$ | Eticheta (clasa) exemplului $d$ | $+1$ = bolnav, $-1$ = sănătos |
| $\mathbf{w}$ | Vectorul de ponderi (weight) | Determină orientarea hiperplanului |
| $b$ | Bias-ul | Deplasează hiperplanul față de origine |
| $n$ | Nr. de instanțe de antrenament | Ex: 100 pacienți |
| $m$ | Nr. de atribute | Ex: 2 (vârstă, greutate) |

### 1.4 Modelul SVM

Funcția de decizie:

$$f(x) = \langle \mathbf{w}, \mathbf{x} \rangle + b$$

Unde $\langle \mathbf{w}, \mathbf{x} \rangle$ este **produsul scalar** dintre $w$ și $x$, adică exact:

$$\langle \mathbf{w}, \mathbf{x} \rangle = w_1 x_1 + w_2 x_2 + \ldots + w_m x_m$$

Deci $f(x) = w_1 x_1 + w_2 x_2 + \ldots + w_m x_m + b$ — similar cu modelul liniar de la regresie, dar folosit pentru **clasificare**.

### 1.5 Regula de clasificare

$$\langle \mathbf{w}, \mathbf{x}_i \rangle + b \geq 0 \Rightarrow y_i = +1 \quad \text{(clasa pozitivă)}$$

$$\langle \mathbf{w}, \mathbf{x}_i \rangle + b < 0 \Rightarrow y_i = -1 \quad \text{(clasa negativă)}$$

> **Exemplu concret:** Avem 2 atribute: $x_1$ = vârstă, $x_2$ = greutate.
> Dacă $w_1 = 0.5$, $w_2 = -0.3$, $b = 1$:
>
> Un pacient cu vârstă=40, greutate=80:
> $f(x) = 0.5 \cdot 40 + (-0.3) \cdot 80 + 1 = 20 - 24 + 1 = -3$
>
> $f(x) = -3 < 0$ → **clasa -1** (sănătos)

---

## 2. Marginea (Margin) și Vectorii Suport

### 2.1 Ce este marginea?

**Marginea** = distanța cea mai scurtă dintre hiperplanul de separare și cele mai apropiate puncte din fiecare clasă.

Pot exista **mai multe hiperplane** care separă corect datele. SVM-ul le alege pe cel cu **marginea cea mai largă** — asta minimizează eroarea de generalizare.

### 2.2 Ce sunt vectorii suport?

**Vectorii suport** = acele puncte din datele de antrenament care cad **exact pe marginea** hiperplanului. Sunt punctele cele mai "dificile" — cele mai apropiate de granița de decizie.

> Dacă elimini un vector suport, hiperplanul se schimbă. Dacă elimini orice alt punct, hiperplanul rămâne la fel.

### 2.3 Ecuațiile marginii

Hiperplanul de decizie: $\langle \mathbf{w}, \mathbf{x} \rangle + b = 0$

Marginile (frontierele):
- $H_+: \langle \mathbf{w}, \mathbf{x} \rangle + b = +1$ (marginea clasei pozitive)
- $H_-: \langle \mathbf{w}, \mathbf{x} \rangle + b = -1$ (marginea clasei negative)

Lățimea marginii:

$$\text{margine} = \frac{2}{\|\mathbf{w}\|}$$

Unde $\|\mathbf{w}\|$ este norma (lungimea) vectorului $w$.

> **Maximizarea marginii** $\frac{2}{\|w\|}$ este echivalentă cu **minimizarea** $\|w\|$ (sau $\|w\|^2$).

---

## 3. Tipuri de Probleme în SVM

### 3.1 Date liniar separabile

Există un hiperplan care separă **perfect** cele două clase, fără nicio eroare.

Se folosește un **Maximal Margin Classifier** — hiperplanul care maximizează marginea.

> **Problemă:** Este foarte sensibil la **outlieri**. Un singur punct amplasat greșit poate schimba dramatic hiperplanul.

### 3.2 Date neliniar separabile (Soft Margin)

În realitate, rareori datele sunt perfect separabile. Soluția: **permitem câteva erori** de clasificare.

Introducem un **Soft Margin** — o margine care tolerează:
- Puncte care cad **în interiorul marginii** (dar pe partea corectă)
- Puncte care sunt **misclassified** (pe partea greșită)

#### Parametrul de penalizare $C$

$C$ controlează **cât de mult penalizăm erorile**:

| $C$ | Efect | Margine | Erori tolerate |
|---|---|---|---|
| $C$ mic | Permite mai multe erori | Largă | Multe |
| $C$ mare | Penalizează puternic erorile | Îngustă | Puține |

> **Exemplu:** $C$ foarte mare → Maximal Margin Classifier (0 erori tolerate, dar sensibil la outlieri).
> $C$ mic → margine largă, câteva erori, dar generalizează mai bine pe date noi.
>
> Valoarea optimă a lui $C$ se determină prin **Cross-Validation**.

### 3.3 Date non-liniar separabile (Kernel Trick)

Uneori, niciun hiperplan liniar nu poate separa datele, nici cu erori.

**Exemplu vizual:** Imaginează-ți un cerc de puncte roșii înconjurat de puncte verzi. Nicio dreaptă nu le poate separa.

**Soluția — Kernel Trick:**

1. Se **transformă datele** într-un spațiu cu **mai multe dimensiuni** (feature space)
2. În noul spațiu, datele **devin liniar separabile**
3. Se găsește hiperplanul de separare în noul spațiu
4. Se **proiectează înapoi** rezultatul în spațiul original

> **Exemplu concret cu date 1D:**
>
> Avem pacienți pe o axă (doza de medicament):
> - Roșu (nevindecat): doze mici și doze mari
> - Verde (vindecat): doze medii
>
> Niciun prag simplu nu poate separa roșu de verde pe o singură axă.
>
> **Transformarea:** Creăm o a doua dimensiune: $x_2 = x_1^2$ (doză la pătrat).
>
> În spațiul 2D ($x_1$, $x_1^2$), punctele roșii și verzi devin separabile printr-o dreaptă!

**Avantajul kernel trick-ului:** Nu calculăm explicit coordonatele în spațiul extins — funcția kernel face calculele eficient.

---

## 4. Funcții Kernel

O funcție kernel calculează **similaritatea** dintre două puncte, ca și cum ar fi fost transformate într-un spațiu cu dimensiuni mai mari, dar **fără a face transformarea explicit**.

### 4.1 Kernele clasice

#### Kernel liniar

$$K(x_1, x_2) = \langle x_1, x_2 \rangle = x_1 \cdot x_2$$

Fără nicio transformare — simplu produs scalar.

> Se folosește când: $m$ (nr. atribute) $\gg$ $n$ (nr. instanțe)

#### Kernel polinomial

$$K(x_1, x_2) = (\langle x_1, x_2 \rangle + 1)^d$$

Unde:
- $\langle x_1, x_2 \rangle$ = produsul scalar al celor două puncte
- $d$ = gradul polinomului (1, 2, 3, ...)
- $+1$ permite și termeni de grad mai mic

> **Exemplu:** Dacă $d = 2$ și avem puncte 2D, kernel-ul polinomial creează implicit atributele $x_1^2$, $x_2^2$, $x_1 x_2$ — un spațiu 3D.
>
> Valoarea optimă a lui $d$ se determină prin **Cross-Validation**.

#### Kernel RBF (Radial Basis Function) / Gaussian

$$K(x_1, x_2) = \exp(-\sigma \|x_1 - x_2\|^2)$$

Unde:
- $\|x_1 - x_2\|^2$ = distanța euclidiană la pătrat dintre cele două puncte
- $\sigma$ = parametru legat de dispersia (spread-ul) datelor
- $\exp$ = funcția exponențială ($e$ la puterea ...)

> RBF-ul proiectează datele într-un spațiu cu **dimensiuni infinite** — și totuși funcționează eficient datorită kernel trick-ului!
>
> Se folosește când: $m$ mare, $n$ mediu. Atributele trebuie **normalizate** la intervalul $[0, 1]$.

### 4.2 Kernele multiple

Se pot combina mai multe kernele:

- **Liniar:** $K(x_1, x_2) = \sum w_i K_i(x_1, x_2)$
- **Neliniar fără coeficienți:** $K = K_1 + K_2 \cdot \exp(K_3)$
- **Neliniar cu coeficienți:** $K = K_1 + c_1 \cdot K_2 \cdot \exp(c_2 + K_3)$

---

## 5. Configurarea unui SVM

### 5.1 Alegerea kernelului — reguli practice

| Situație | Kernel recomandat | De ce |
|---|---|---|
| $m \gg n$ (multe atribute, puține instanțe) | Liniar | Spațiul e deja suficient de mare |
| $m$ mare, $n$ mediu | RBF / Gaussian | Trebuie separare neliniară |
| $m$ mic, $n$ mare | Se adaugă atribute, apoi liniar | Spațiul original e prea simplu |

### 5.2 Normalizarea

Atributele trebuie **normalizate** (scalate) la intervalul $[0, 1]$ înainte de antrenare, altfel atributele cu valori mari domină calculul distanțelor.

### 5.3 Algoritmul de antrenare: SMO

**Sequential Minimal Optimization** — algoritmul standard de antrenare pentru SVM. Rezolvă problema de optimizare intern (nu trebuie să-l implementezi manual).

---

## 6. Clasificare Multiclasă

SVM-ul nativ face doar **clasificare binară** (2 clase). Pentru mai multe clase se folosește strategia:

### One-vs-Rest (Unu contra restul)

Dacă avem $k$ clase, antrenăm **$k$ SVM-uri separate**:

- SVM 1: Clasa A vs. (B + C)
- SVM 2: Clasa B vs. (A + C)
- SVM 3: Clasa C vs. (A + B)

La clasificare, se aplică toate cele $k$ SVM-uri și se alege clasa cu scorul cel mai mare.

> **Exemplu:** Clasificare animale (Cat / Dog / Panda) cu 3 SVM-uri:
> 1. Cat vs. (Dog + Panda) → scor 0.8
> 2. Dog vs. (Cat + Panda) → scor 0.3
> 3. Panda vs. (Cat + Dog) → scor 0.1
> → Clasificare: **Cat** (scorul cel mai mare)

---

## 7. Avantaje și Dificultăți SVM

### 7.1 Avantaje

1. **Pot lucra cu orice fel de date** — liniar separabile sau nu, distribuție uniformă sau nu
2. **Funcțiile kernel** creează automat noi atribute (similar cu straturile ascunse din rețelele neuronale)
3. **Dacă problema e convexă** → oferă **soluție unică** (optimul global), spre deosebire de RNA care pot cădea în optime locale
4. **Selectează automat** dimensiunea modelului (prin vectorii suport) — nu trebuie configurat manual ca straturile din RNA
5. **Nu învață pe derost** (overfitting) — RNA se confruntă cu overfitting chiar și cu cross-validare

### 7.2 Dificultăți

1. Lucrează doar cu **atribute reale** (numerice)
2. Nativ doar **clasificare binară** (necesită One-vs-Rest pentru multiclasă)
3. **Background matematic dificil**

### 7.3 Tool-uri

- **LibSVM** — cel mai popular (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- **Weka** → algoritmul SMO
- **SVMLight**, **SVMTorch**

---

## 8. SVM Structurate

Extindere pentru sarcini complexe cu **ieșiri structurate**:

- **Intrări:** orice tip (text, imagine, structură moleculară)
- **Ieșiri:** structuri complexe (arbore sintactic, graf, semnal, propoziție)

---

## 9. Evoluția conceptuală: De la margine maximă la kernel

Rezumând drumul conceptual din curs:

1. **Maximal Margin Classifier** — funcționează perfect pe date liniar separabile, dar e sensibil la outlieri
2. **Support Vector Classifier (Soft Margin)** — permite erori controlate prin parametrul $C$, mai robust
3. **Support Vector Machine (Kernel Trick)** — transformă datele non-liniar separabile în liniar separabile prin funcții kernel

---

## 10. ⚠️ Aspecte Critice pentru Examen

### Tipare din SUBIECT_EXAMEN_2025

Examenul nu conține un subiect explicit pe SVM, dar conceptele sunt necesare pentru:

**Subiect 4** — *"Câți coeficienți trebuie optimizați într-o rețea neuronală?"*
→ Compară cu SVM: la SVM coeficienții sunt $w$ și $b$, la RNA sunt ponderile + bias-urile tuturor straturilor

**Concepte generale** cerute la examen care implică SVM:
→ Ce este un hiperplan de separare
→ Ce sunt funcțiile kernel
→ Diferența dintre clasificare binară și multiclasă

### Concepte-cheie de memorat

1. **Hiperplan:** $\langle w, x \rangle + b = 0$ — separatorul liniar
2. **Clasificare:** $f(x) \geq 0 \Rightarrow +1$, $f(x) < 0 \Rightarrow -1$
3. **Margine** = $\frac{2}{\|w\|}$ — se maximizează
4. **Vectori suport** = punctele cele mai apropiate de hiperplan (pe margine)
5. **Soft Margin** = permitem erori, controlate de $C$
6. **Kernel trick** = transformăm datele în spațiu cu dimensiuni mai mari
7. **Kernel polinomial:** $K = (\langle x_1, x_2 \rangle + 1)^d$ — gradul $d$ se alege prin CV
8. **Kernel RBF:** $K = \exp(-\sigma \|x_1 - x_2\|^2)$ — $\sigma$ = dispersia datelor
9. **Multiclasă** = One-vs-Rest: $k$ SVM-uri pentru $k$ clase
10. **Avantaj cheie vs RNA:** SVM oferă optimul global, RNA pot cădea în optime locale

---

*Rezumat generat exclusiv din `03_ML_SVM.pdf` și `3_1_Masini_cu_suport_vectorial__MSV_.md`*
