# GNNs and Generative Models for Drug Discovery

These notes were made by Logan Mondal Bhamidipaty as part of a tutorial completed at Oxford (Michaelmas 2023) under Martin Buttenschoen.

# Graph Neural Networks (GNNs)

## Basics

**Definition:** _A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances)._

**Tasks:** In general, there are 3 kinds of tasks that GNN perform: node-level (e.g., predicting relations in social media networks), edge-level (image scene understanding), and graph-level (molecule property prediction). 

**Representation of Graphs:** GNNs often have to handle large data (e.g., social media network), so it's important to have efficient representations. Adjacency matrices are often large and sparse, so adjacency lists are frequently preferred. Another bonus of adjacency lists is that they are _permutation invariant_.

As will be briefly touched on later, GNNs can be viewed as a generalization of many types of neural networks including transformers and CNNs.

## Message-Passing

Message passing allows graphs to build latent representations of graph features that incorporate connectivity information (e.g., neighbors).

**Simple Generalization:** Simple message passing involves three steps:
1. Aggregating information from "nearby" nodes
2. Pooling this information (e.g., using sums, averages, or attention)
3. Passing pooled information through an update function (often a neural network)

In this way, a latent representation of a graph is built up over time that gradually incorporates the connectivity information of a graph. 

**Graph Convolutions, Laplacians, and Virtual Augments**: One common way of aggregating information is with _graph convolutions_ that harness the many interesting properties of [graph Laplacians](https://en.wikipedia.org/wiki/Laplacian_matrix). In special cases where graphs are sparsely connected or have orphan nodes, researchers might artificially add a "virtual node" or "virtual edge" that connects all of targeted objects. 

## "Flavors" of GNN Layers

_Reference: [[2104.13478] Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478), Sec: 5.3 Graph Neural Networks_

Following the reference above, we can divide GNN layers into three distinct flavors: (1) convolutional, (2) attentional, and (3) message-passing. Where each subsequent category is a generalization of the previous so that $\text{convolution } \subseteq \text{ attention } \subseteq \text{ message-passing}$. 

