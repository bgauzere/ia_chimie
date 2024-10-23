---
marp: true
title: Historique de l'Intelligence Artificielle
theme: default
paginate: true
---

# Introduction to Machine Learning

---

# What is AI ? and ML ?

## Artificial Intelligence

- **AI**: The field of study that tries to make computers "smart"

- **ML**: A subset of AI that uses statistical methods to enable computers to learn from data

- **Deep Learning**: A subset of ML that uses neural networks with many layers


---

# A Brief History of AI

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

![center](figures/timeline.svg)

---
# Darthmouth Conference 

> _"We propose that a 2-month, 10-man study of artificial intelligence be carried out during the summer of 1956 at Dartmouth College in Hanover, New Hampshire. The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it. An attempt will be made to find how to make machines use language, form abstractions and concepts, solve kinds of problems now reserved for humans, and improve themselves. We think that a significant advance can be made in one or more of these problems if a carefully selected group of scientists work on it together for a summer."_ 

---

# How AI evolved through time ?

## Theory : The imitation game

![center fit](figures/Turing_test_diagram.png)

---

## Expert systems

![bg right fit](figures/simpsons.jpeg)

```prolog
person(homer).
person(marge).
person(bart).

parent(homer,bart).
parent(marge,bart).

grand_parent(X,Y) :- parent(X,Z), parent(Z,Y).
```

---

## Statistical learning, learning by example
![bg right width:600](figures/intro_supervised_learning.svg)

---

## Representation Learning

![bg right fit](figures/perceptron.svg)
[Tensorflow playground](https://playground.tensorflow.org/)

---

# AI Examples and Applications


---

# Go

![center width:1000](./figures/alpha_go.jpg)

---

# Medical Imaging 

![bg right fit](./figures/ia_medical.png)


---

# Autonomous Driving

![bg left fit](./figures/rouen_car.jpg)

---

# Recommandation Systems 

![bg right fit](./figures/recommandation.svg)

---

# Spam Detection

![bg left fit](./figures/spam.png)

---

# Object Detection

![bg right fit](./figures/object_detection.png)

---

# Generative Models

![bg left fit](./figures/dalle2.png)

---

# NLP Generation

![bg right fit](./figures/chat_gpt.png)

---

# And Chemical Science

## Molecular Property Prediction

![center width:800](figures/ml_qspr.png)

---

## Molecular Discovery

![center width:800](figures/nature_antibiotics.png)

---
## And even Nobel prizes !

![width:400 left](figures/chemistry_nobel_prize.png)

![bg right fit](figures/alphafold.png)

---

# General Problem

## What we are talking about?
![bg right width:600](figures/intro_supervised_learning.svg)

- Learning through examples
- Mimic human tasks (AI?)
- Produce outputs given some inputs (functions?)

---

## Supervised Learning
 **Purpose**  
Given a dataset $\{  (x_i, y_i) \in \mathcal{X} \times \mathcal{Y}, i = 1, \dots, N \}$, learn the dependencies between $\mathcal{X}$ and $\mathcal{Y}$.

- Example: Learn the relationship between cardiac risk and food habits. $\x_i$ is a person described by $d$ features about their food habits; $y_i$ is a binary category (risky, not risky).
- **$y_i$ is essential for the learning process.**
- Methods: K-Nearest Neighbors, SVM, Decision Tree, etc.

---

## How to Encode Data

The Matrix data : $X \in \mathbb{R}^{n \times p}$

### Samples
- $n$ samples (number of rows)
- $X(i,:) = x_i^\top$ : the i-th sample
- $x_i \in \mathbb{R}^{p}$

### Features
- $p$ features (number of columns)
- Each sample is described by $p$ features
- $X(:,j)$ : the j-th feature for all samples

$$X(i,j) : \text{j-th feature of the i-th sample.}$$

---

![X Data Visualization](figures/X_data.svg)

---

## Learning Model
### Model
$$
f : \mathcal{X} \to \mathcal{Y}
$$
$$
x_i \to \hat{y}
$$

We want that:
$$
f(x_i) \simeq y_i
$$

---

## Example
![bg right fit](figures/data.svg)

What is the underlying $f$ function?

---

![center fit ](figures/interpolation.svg)

---

## How to find a good $f$?
$$
f^\star = \arg \min_f \mathcal{L}(f(X), y) + \lambda \Omega(f)
$$

- $\mathcal{L} : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$
- $\lambda \in \mathbb{R}^+$
- $\Omega : (X \to Y) \to \mathbb{R}^+$

---

## Fit to Data Term
$$
\mathcal{L}(f(X), y)
$$

- Ensures the model fits the data
- Penalizes when the predicted value $f(x_i)$ is far from $y_i$

---

## Regularization Term
$$
\Omega(f)
$$

- Constrains the complexity of function $f$
- Occam's razor: simpler is better
- $\lambda$: Weights the balance between the two terms

---

![center](figures/non_interpolation.svg)
- Very high $\mathcal{L}(f(X), y)$
- Very low $\Omega(f)$

---

![center](figures/linear_interpolation.svg)
- High $\mathcal{L}(f(X), y)$
- Low $\Omega(f)$

---

![center](figures/squared_interpolation.svg)
- $\mathcal{L}(f(X), y)=0$
- High $\Omega(f)$

---

![center](figures/infinite_interpolation.svg)
- $\mathcal{L}(f(X), y)=0$
- Very high $\Omega(f)$

---

## Generalization
*Good models generalize well*

- Good generalization: good prediction on unseen data
- Hard to evaluate without bias
- Overfitting
- Regularization term prevents overfitting

---

# Regression and Classification

## Binary Classification
- $\mathcal{Y} = \{0,1\}$
- Dog or cat? Positive or Negative?
- Performance: Accuracy, recall, precision, etc.

![center](figures/classifier.svg)

---

## Regression
- $\mathcal{Y} = \mathbb{R}$
- Stock market, house price, boiling points of molecules, etc.
- Performance: RMSE, MSE, MAE, etc.

![center](figures/regressor.svg)

---

## And Many Others
- Ranking
- MultiClass classification
- Multi Labeling
- etc.

---

# Methods
## Machine Learning Methods for Classification

- K-nearest neighbors
- Random forests
- SVM & consorts
- Multi-Layer Perceptron

---

# A first approach KNN

---

# k Nearest Neighbors
Determine properties from similar data

![center](figures/knn_1.svg)

---

# k Nearest Neighbors
Determine properties from similar data

![center](figures/knn_2.svg)

---
# k Nearest Neighbors
Determine properties from similar data

![center](figures/knn_3.svg)

---
# k Nearest Neighbors
Determine properties from similar data

![center](figures/knn_4.svg) 

---

## K-NN Hyperparameters
### Number of Neighbors $k$
- Low number: high variability, high accuracy
- High number: smoother

### Distance
- Similarity depends on data, structure, task
- Euclidean, Manhattan, ad hoc distances

---

## KNN: The (Simple) Code!

```python
from sklearn.neighbors import KNeighborsClassifier
k = 5
metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
knn.fit(X, y)  # Training
y_pred = knn.predict(X)  # Prediction
```
-  [Metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html)

- [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
