# SolutionSampler: Core Ideas, Graph Construction, and Algorithm

## 1. Overview
This document summarizes the essential components needed to construct a neural solution sampler for the **Vertex-Guard Art Gallery Problem (AGP)** using the triangulation–visibility–hypergraph reduction and the optimization framework inspired by **Caramanis et al. (2023)**.

The goal is to convert the continuous geometric guarding problem into a **finite, discrete, differentiable**, and **policy-gradient-optimizable** form.

---

## 2. Crucial Ideas

### **2.1 Exact Discretization for Vertex Guards**
A simple polygon $P$ with vertex set $V$ is triangulated using only original polygon vertices. The triangulation does **not** add new guard candidates. Every point in the polygon lies in exactly one triangle.

Key property:
- A vertex guard sees an entire triangle **iff** it sees all three triangle vertices.
- Therefore, guarding all triangles is equivalent to guarding the entire polygon.

This establishes an **exact equivalence** between:
- guarding the polygon, and
- hitting all triangles via visible vertices.

### **2.2 Hypergraph Representation**
For each triangle $t_i = (a_i,b_i,c_i)$, define:
- **Triangle guard set** $G_i = \{v \in V : v \text{ sees } a_i, b_i, c_i\}$.
- Build hypergraph $H = (V, \{G_i\})$.

Then the classical minimum vertex guard problem is exactly equivalent to the minimum hitting set on this hypergraph.

This compresses continuous geometry to a **polynomial-sized binary incidence matrix** $M$.

### **2.3 Compatibility with NCO**
The incidence matrix $M$ provides an instance representation $\psi_I(I)$. A selection vector $s \in \{0,1\}^n$ provides the solution representation $\psi_S(s)$.

Coverage is linear:
$$
(Ms)_i = \sum_{v \in G_i} s_v.
$$

A smooth surrogate cost:
$$
L(s;I) = \lambda \|s\|_1 + (1-\lambda) \sum_i \sigma(1 - (Ms)_i)
$$
fits the **bilinear/smooth structure** required by Caramanis et al. for benign landscape properties.

---

## 3. Graph Construction Pipeline

### **Step 1 — Input**
Polygon $P$ with vertex set $V = \{v_1, …, v_n\}$.

### **Step 2 — Triangulation**
- Triangulate $P$ using only polygon vertices.
- Produce triangles $T = \{t_1, …, t_{n-2}\}$.

### **Step 3 — Visibility Graph $G_V$**
Construct edge $(u,v)$ if segment $uv$ lies inside polygon.

### **Step 4 — Build Hyperedges**
For each triangle:
- Compute vertices that see all triangle vertices.
- Add hyperedge $G_i$.

### **Step 5 — Build Incidence Matrix**
Size:
- Rows: triangles.
- Columns: vertices.

Entry:
$$M[i,v] = 1 \text{ iff } v \in G_i.$$

This matrix fully encodes the guarding instance.

---

## 4. Solution Sampler (Algorithm)

### **Step 1 — Policy Parameterization**
Define a neural policy $p_\theta(s)$ over binary guard vectors:
- independent Bernoulli logits, or
- GNN/Transformer for correlated selections.

### **Step 2 — Sample Solutions**
Draw $K$ samples $s^{(1)}, …, s^{(K)} \sim p_\theta$.
Each sample represents a proposed guard set.

### **Step 3 — Soft Coverage Evaluation**
Compute coverage:
$$
c_i^{(k)} = (Ms^{(k)})_i.
$$

### **Step 4 — Loss Computation**
Smooth objective:
$$
L(s^{(k)};I) = \lambda \|s^{(k)}\|_1 + (1-\lambda)\sum_i \sigma(1 - c_i^{(k)}).
$$

### **Step 5 — Policy Gradient Update**
$$
\nabla_\theta J = \frac{1}{K} \sum_k (L(s^{(k)}) - b) \, \nabla_\theta \log p_\theta(s^{(k)}).
$$
Where:
- $b$: moving baseline,
- entropy regularization ensures exploration.

Update:$$
\theta \leftarrow \theta - \eta \nabla_\theta J.
$$

### **Step 6 — Return Sampler**
After training, $p_\theta$ acts as a **solution generator** for guard sets.

---

## 5. Key Properties
- **Exactness**: the hypergraph formulation captures the true vertex-guard problem.
- **Compressibility**: continuous geometry becomes a finite incidence matrix.
- **Optimizable**: cost is smooth + bilinear, satisfying Caramanis theoretical assumptions.
- **Modular**: any differentiable policy model can be plugged in.

---

## 6. Output
A trained sampler that produces high-quality guard sets, approximating the minimum vertex guard solution.

---

End of file.

