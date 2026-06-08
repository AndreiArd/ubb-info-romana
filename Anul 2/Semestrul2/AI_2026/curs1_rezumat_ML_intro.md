# Rezumat Curs 1 — Introducere în Machine Learning & Sisteme Inteligente
**Surse:** `01_ML_intro.pdf` + `1__Sisteme_inteligente.md`
**UBB Informatică Anul 2 — Inteligență Artificială**

---

## 1. Ce este Machine Learning (ML)?

Machine Learning este o ramură a Inteligenței Artificiale care se ocupă cu crearea de sisteme capabile să **învețe singure din date**, fără a fi programate explicit pentru fiecare sarcină.

### Definiții relevante
- **Arthur Samuel (1959):** "field of study that gives computers the ability to learn without being explicitly programmed"
- **Tom Mitchell (1998):** un program învață din experiența **E** față de sarcina **T** cu măsura de performanță **P**, dacă performanța sa pe **T** (măsurată prin **P**) se îmbunătățește cu experiența **E**.
- **Yoshua Bengio et al. (2016):** "studiul algoritmilor care se îmbunătățesc prin experiență și extrag tipare din date pentru a face predicții sau decizii"

### Programare tradițională vs. Machine Learning

| Aspect | Programare tradițională | Machine Learning |
|--------|------------------------|-----------------|
| Input  | Date + Reguli → Calculator | Date + Răspunsuri corecte → Calculator |
| Output | Răspunsuri | **Reguli (model)** |
| Adaptare | Manuală | Automată |
| Exemple | Calculatoare de buzunar | Filtre spam, recunoaștere imagini |

> **Esența ML:** În programarea clasică, programatorul scrie regulile. În ML, **calculatorul descoperă singur regulile** din exemple.

---

## 2. Proiectarea unui Sistem de Învățare Automată

Orice sistem ML este proiectat urmând un framework cu trei întrebări fundamentale:

1. **Ce modele de predicție sunt posibile?** → Ipotezele care se validează
2. **Cât de bun este un model?** → Funcția de cost (loss function)
3. **Cum se identifică cel mai bun predictor?** → Algoritmul de optimizare

### Pașii proiectării

1. **Stabilirea scopului** — Ce vrem să învețe sistemul? (clasificare, regresie, clustering etc.)
2. **Alegerea algoritmului de învățare** — Algoritmul induce ipoteze care:
   - se potrivesc cu datele de antrenament
   - generalizează cât mai bine pe date nevăzute (date de test)
3. **Definirea funcției de cost** — Minimizarea erorii:
   - Eroarea de predicție (distanța valorilor prezise față de valorile reale)
   - Eroarea de clasificare (câte exemple au fost clasificate greșit)
4. **Evaluarea sistemului** — Experimental (cross-validare, acuratețe) sau teoretic (complexitate computațională)

---

## 3. Tipologia Sistemelor de Învățare Automată

### A. După experiența din timpul învățării

#### 3.1 Învățare Supervizată
- **Definiție:** Datele de antrenament sunt **etichetate** (se cunosc perechile intrare–ieșire)
- **Scop:** Să se determine o funcție $f: \text{atribute} \to \text{ieșire}$ care generalizează pe date noi
- **Feedback:** Direct — algoritmul se adaptează la perechile cunoscute
- **Notații formale:**
  - Date de antrenament: $\{(\text{atribute}_i, \text{ieșire}_i)\}_{i=1}^{N}$
  - Date de test: $\{(\text{atribute}_i)\}_{i=1}^{n}$
- **Tipuri de probleme:**
  - **Clasificare** — ieșirea este o categorie dintr-o mulțime finită (ex: spam/non-spam, diagnostic medical)
  - **Regresie** — ieșirea este un număr real (ex: prețul unui imobil)
- **Exemple:** regresie liniară, SVM, rețele neuronale, arbori de decizie

#### 3.2 Învățare Nesupervizată
- **Definiție:** Datele de antrenament **NU sunt etichetate** — nu se cunosc ieșirile
- **Scop:** Descoperirea de structuri/tipare ascunse în date
- **Feedback:** Inexistent (nu există ieșiri de referință)
- **Exemple de probleme:**
  - **Clustering (clusterizare)** — gruparea datelor similare (ex: zone cu cazuri COVID concentrate)
  - **Reducerea dimensionalității** — comprimarea datelor păstrând informația esențială
- **Algoritmi:** k-Means, PCA

#### 3.3 Învățare cu Întărire (Reinforcement Learning)
- **Definiție:** Un agent interacționează cu un mediu și primește **recompense/penalizări** pentru acțiuni
- **Scop:** Învățarea unui model optim de acțiune (politică)
- **Caracteristici:**
  - Predicție de secvențe de decizii/acțiuni
  - Sistem de recompense pentru fiecare decizie
  - Datele sunt **secvențiale și corelate** (nu i.i.d. ca în supervizat)
- **Exemple:** AlphaGo, jocuri video, roboți autonomi, planificare

#### 3.4 Tabelul comparativ complet

| Aspect | Supervizat | Nesupervizat | Cu întărire |
|--------|-----------|--------------|-------------|
| Date | Etichetate | Neetichetate | Secvențiale |
| Feedback | Instructiv | Absent | Evaluativ (recompensă) |
| Scop | Predicție | Descoperire structuri | Acțiune optimă |
| Exemplu | Clasificare email | Clustering clienți | AlphaGo |

### B. După modelul învățat (algoritmul)
- Rețele neuronale artificiale
- Mașini cu suport vectorial (MSV/SVM)
- Algoritmi evolutivi
- k-Nearest Neighbors (kNN)
- Arbori de decizie
- Modele Markov ascunse

---

## 4. Tipuri de Scopuri / Probleme în ML

### 4.1 Clasificare
Maparea unei intrări la una dintre $k$ **clase discrete predefinite**.

**Formal:** Se dau $(x_i, y_i)_{i=1}^N$ unde $y_i \in \{C_1, C_2, \ldots, C_k\}$. Se învață $f(x) \to C_j$.

**Exemplu:** Detectarea tumorilor (vasculară / nevasculară / angiogenă), detectarea COVID din RMN.

### 4.2 Regresie
Maparea unei intrări la un **număr real continuu**.

**Formal:** Se dă $(x_i, y_i)_{i=1}^N$ unde $y_i \in \mathbb{R}$. Se învață $f(x) \approx y$.

**Exemplu:** Predicția prețului unui imobil pe baza numărului de camere, vechimii, suprafeței.

### 4.3 Clusterizare (Clustering)
Gruparea datelor **fără etichete** în clustere omogene.

**Exemplu:** Identificarea zonelor geografice cu concentrare mare de cazuri COVID.

### 4.4 Planificare
Generarea unei **secvențe de acțiuni** optime pentru atingerea unui scop.

**Exemplu:** Un robot care găsește drumul optim de la punctul A la punctul B.

### 4.5 Regresie simbolică
Descoperirea **formei funcției matematice** care descrie datele.

**Exemplu:** Găsirea formulei curbei care aproximează un contur geografic.

---

## 5. Calitatea Învățării & Evaluarea Modelelor

### 5.1 Metrici de performanță
**Acuratețea** (Accuracy):

$$Acc = \frac{\text{nr. exemple corect clasificate}}{\text{nr. total de exemple}}$$

Calculată atât în **faza de antrenare**, cât și în **faza de testare**.

### 5.2 Metode de evaluare

#### Seturi disjuncte antrenare/test
- Se împarte dataset-ul în: **set de antrenare** (+ opțional set de validare) + **set de testare**
- Parametrii modelului se estimează pe setul de antrenare; cei mai buni parametri (obținuți pe validare) se folosesc pentru modelul final
- **Recomandat pentru:** seturi de date mari

#### Validare încrucișată (k-fold Cross-Validation)
- Datele de antrenament se împart în **h subseturi egale**
- La fiecare iterație: (h-1) subseturi → antrenare, 1 subset → validare
- Performanța finală = media pe cele h rulări
- **Recomandat pentru:** seturi de date mici (tipic h = 5 sau h = 10)

$$CV_{(h)} = \frac{1}{h} \sum_{i=1}^{h} \text{Error}_i$$

#### Leave-One-Out Cross-Validation (LOOCV)
- Caz particular: h = numărul total de exemple
- La fiecare iterație, un singur exemplu este lăsat pentru validare
- **Recomandat pentru:** seturi de date foarte mici

### 5.3 Dificultăți în Învățare

#### Overfitting (Învățare pe derost / Supra-potrivire)
- **Definiție:** Modelul se potrivește prea accentuat cu datele de antrenament, dar **nu generalizează** pe date noi
- **Simptom:** Performanță bună pe antrenament, **performanță slabă pe test**
- **Cauze:** Model prea complex, prea puține date
- **Soluții:** Regularizare, reducerea complexității modelului, mai multe date, cross-validare

#### Underfitting (Sub-potrivire)
- **Definiție:** Modelul este prea simplu și **nu captează tiparele** din date
- **Simptom:** Performanță slabă **atât pe antrenament cât și pe test**
- **Soluții:** Model mai complex, mai multe feature-uri, mai multe iterații de antrenare

---

## 6. Principiul de Bază al Algoritmilor ML

Orice algoritm de ML are ca principiu fundamental:

$$\text{Minimizarea unei funcții de cost (loss function) pe datele de antrenament}$$

Tipuri de erori minimizate:
- **Eroarea de predicție** — distanța dintre valorile prezise și valorile reale
- **Eroarea de clasificare** — numărul de exemple clasificate greșit
- **Eroarea structurală** — cât de neomogene sunt structurile produse (în clusterizare)

---

## 7. ⚠️ Aspecte Critice pentru Examen

### Tipare identificate în SUBIECT_EXAMEN_2025

**Întrebare tip 1 — Citirea unui grafic de regresie:**
> *"Se dă un model liniar. Precizați prețul previzionat pentru X."*
→ Necesită înțelegerea conceptului de regresie (Curs 1) + citire de grafic

**Întrebare tip 2 — Compararea a două modele:**
> *"Două modele diferite, care are eroarea medie pătratică mai mare?"*
→ Înțelegerea funcției de cost și a conceptului de eroare (Curs 1 + Curs 2)

**Întrebare tip 5 — Concepte LLM:**
> *"GPT-3 este un LLM? ChatGPT este un LLM?"*
→ Necesită fundamente clare de tipologie ML din Curs 1

### Concepte cheie de memorat
1. Diferența supervizat / nesupervizat / cu întărire
2. Clasificare vs. regresie (tip de ieșire!)
3. Overfitting vs. underfitting (simptome și soluții)
4. Cross-validare — când se folosește și cum funcționează
5. Funcția de cost = măsura calității modelului
6. Framework proiectare ML: ipoteze → funcție cost → algoritm de optimizare

---

*Rezumat generat exclusiv din `01_ML_intro.pdf` și `1__Sisteme_inteligente.md`*
