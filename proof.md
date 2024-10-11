# Proof of Thesis

## Preliminaries

### Data Restoration: Problem Definition

For a 2D time dependent PDE. We have PDE data $ D \sub R^{2+1} $ which represents $Height\times Width\times Time$, $D'$ is the filled corrupted data in a neighbor of $D$, i.e. $D' \in \mathcal{B}(D,r) $. Denote $\Omega = \mathcal{B}(D,r) $. Suppose we have a series of continues function $\{f_i\}^\infty_{i=1}，f_i(D)=D$, and suqare error function $\mathcal{l}$

$$
\forall\ M_1,\ M_2 \in \Omega,\ \mathcal{l}(M_1,M_2)=\sum_{i,j}(m^1_{ij}-m^2_{ij})^2
$$

We can easily verify that $(\Omega,\ \mathcal{l})$ is a Metric Space w.r.t distance $L$.


## Proof 1: Converge of Recovery map
Let the function series $\{f_i\}^\infty_{i=1}$ defined as above. $\forall D' \in \Omega, f_i(D')\in \Omega$. Assume we have

$$
\mathcal{l}(f_{i+1}(D'),D)<\mathcal{l}(f_{i}(D'),D)
$$

then $a_i=\{\mathcal{l}(f_{i}(D'),D)\}$ monotone decreasing, and $a_n\geq 0$ because $a_n$ is a distance function. According to the monotone bounded convergence theorem, series $a_n$ has a finite limit.

Define the distance in $\mathcal{L}(\Omega, \Omega)$, 
$d(f_i,f_j)=(\int_\Omega (f_i(x)-f_j(x))^2 dx)^{\frac12}$. 
It can be easily verified that the distance satisfy $d(af_i,af_j)=ad(f_i,f_j)$ and $d(f_i-f_j,0)=d(f_i,f_j)$. So the Metrc Space $(\mathcal{L}(\Omega, \Omega),\ d)$ is also a Normed Space. 

Then $\forall D' \in \Omega$, we have
$$

\mathcal{l}(f_i(D'),f_j(D'))\leq \mathcal{l}(f_i(D'),f_i(D)) + \mathcal{l}(f_i(D),f_j(D)) + \mathcal{l}(f_j(D),f_j(D'))

$$
the middle term $\mathcal{l}(f_i(D),f_j(D)) = \mathcal{l}(D,D) = 0$ according to the definition of $\{f_i\}^\infty_{i=1}$. And we have that
$$

\forall \epsilon \geq 0, \ \exist N>0:\ \forall i,j>N, \mathcal{l}(f_i(D'),D), \mathcal{l}(f_j(D'),D)<\epsilon/2

$$
which yields to $\mathcal{l}(f_i(D'),f_j(D'))<\epsilon $ for all $D' \in \Omega $. This further illustrates $f_i$ is a Cauchy series in Normed Space $(\mathcal{L}(\Omega, \Omega),\ d)$.

In this situation, $f_i$ can be seen as operators in $\Omega$. Because of their continuity and the compactness of $\Omega$, $f_i$ are bounded operators in $\Omega$. Furtherly, the distance and norm still holds in this space, so the Normed Space above changes into $(\mathcal{B}(\Omega, \Omega),\ d)$. Since $\Omega$ is the sub space of $R^3$, then it is complete, which indicate the operator space $\mathcal{B}(\Omega,\Omega)$ is a Banach space.

Finally we get that $\{f_i\}^\infty_{i=1}$ is converge in $\mathcal{B}(\Omega,\Omega)$, which is the desired result. We denote the limit of $\{f_i\}^\infty_{i=1}$ as $f$.

## Lemma 1: Dini Theorem
Let $K$ be a compact metric space. Let $f : K \rightarrow \mathbb{R} $ be a continuous function and $f_n : K \rightarrow \mathbb{R}, n \in \mathbb{N}$, be a sequence of continuous functions. If $\{f_n\}_{n\in \mathbb{N}}$ converges pointwise to $f$ and if
$$

f_n(x)≥f_{n+1}(x)\ \forall x\in\mathbb{K}\ and\ \forall n \in \mathbb{N}

$$
then $\{f_n\}_{n\in \mathbb{N}}$ converges uniformly to $f$.

## Proof 2: Continuity of limit

In Proof 1, we proof $\{f_i\}^\infty_{i=1}$ pointwise converge to $f$. Since $\Omega$ is a compact set. Consider series $\{f-f_i\}^\infty_{i=1}$. 
For every singal point $D_i \in \Omega$, $\{(f-f_i)(D_i)\}$ monotone decreasing and pointwise converge to 0. By Lemma 1, $\{f-f_i\}^\infty_{i=1}$ uniform converge to 0.

So for all $\epsilon > 0, \exist \delta>0, \forall D' \in B(D,\delta)$,
$$

\mathcal{l}(f(D),f(D')) \leq \mathcal{l}(f(D),f_N(D)) + \mathcal{l}(f_N(D),f_N(D')) + \mathcal{l}(f_N(D'),f(D'))

$$
The first and third term tends to 0 because of uniform convergence of $\{f-f_i\}^\infty_{i=1}$. And the second term tends to zero because of the continuity of $f_N$。

## Proof 3: Stability of Recovery map

For a specific turbulence $\delta$ to $f$, the perturbed function is $(f+\delta)$. Assume $\delta$ satisfy $|\delta_i(D)| < \frac{\epsilon}{2},\ \forall D \in \Omega$, then we have
$$

|(f_i + \delta_i)(D) - f_i(D)| = |\delta_i(D)| < \frac{\epsilon}{2}

$$
By triangle inequality
$$

|f(x) - (f_i + \delta_i)(D)| \leq |f(D) - f_i(D)| + |f_i(D) - (f_i + \delta_i)(D)| = |f(D) - f_i(D)| + |\delta_i(D)|

$$
notice that $\{f-f_i\}^\infty_{i=1}$ uniform converge to 0. Then we have

$$
|f(D) - (f_i + \delta_i)(D)| < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon
$$

which indcate the desired result that $\forall \epsilon>0,\ \exist N$, when $i \geq N$ and $|\delta_i(D)| < \epsilon /2,\ \forall D \in \Omega$. It has $ |f(D) - (f_i + \delta_i)(D)| < \epsilon $

## Proof 4: Universial Approximation Theorem for Convolutional Neural Network ??
