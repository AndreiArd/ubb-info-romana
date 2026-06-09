# LUCRURI IMPORTANTE DIN TOATE

# 01_ML_intro.pdf — Introducere în ML / Sisteme Inteligente**  

### Concepte cheie de memorat
1. Diferența supervizat / nesupervizat / cu întărire
2. Clasificare vs. regresie (tip de ieșire!)
3. Overfitting vs. underfitting (simptome și soluții)
4. Cross-validare — când se folosește și cum funcționează
5. Funcția de cost = măsura calității modelului
6. Framework proiectare ML: ipoteze → funcție cost → algoritm de optimizare



# 02_ML_LS_GD.pdf — Least Squares & Gradient Descent

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

### Concepte-cheie de memorat

1. **Formula normală:** $w = (X^T X)^{-1} X^T y$ — soluție directă, fără iterații
2. **Formula GD:** $w_k := w_k - \eta \cdot \text{error} \cdot x_k$ — actualizare iterativă
3. **Eroarea în GD:** $\text{error} = \text{predicție} - \text{realitate}$ (cu semn!)
4. **Sigmoid:** $S(z) = \frac{1}{1+e^{-z}}$, mapează $\mathbb{R} \to (0,1)$
5. **Diferența GD vs SGD:** update după toate exemplele vs. după fiecare exemplu
6. **De ce sigmoid la regresie logistică:** pentru că ieșirea trebuie să fie o probabilitate între 0 și 1
7. **MSE vs SSE:** MSE = SSE/N — ambele au același minim