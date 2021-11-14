
# Reference-Based Deep Metric Learning for Remaining Useful Lifetime Prediction of Printing Screen

## Content

1. Process introduction and motivation
2. State-of-the-Art
3. Methods with proposed solutions
4. Results

## Process introduction and Motivation

The screen printing process has seen growth in manufacturing processes such as decorative stripes on cosmetics and syringes. In real production of syringes, the quality of the printed image will be worse and worse, until the printing screen is replaced, in other words, every printing screen has a useful lifetime.
Using conventional maintenance methods, the printing screen is replaced periodically to avoid bad printed image. Unfortunately, such methods are inefficient due to the possibility of further using it in a case where the printing screen is still in a good condition. This has led to inefficiencies and a lot of waste.

On the other hand, assessing the condition of printing screen is a complex task and the quality of the screen cannot be measured directly and the printed image varies
depending on the current layout. Therefore, in order to assess the printed screen, we use the resulting printed image to assess it, rather than the printed screen itself.

Since the quality of printed image get worse after each use until printing screen is replaced,
meanwhile, the degradation metric is unknown, in this work, neural networks are used
to learn a reference-based degradation metric implicitly.

## State-of-the-Art

1. ResNet
2. Siamese Network
3. Deep Metric Learning
4. Bayesian Optimization
5. Learning Rate Test & One Cycle Policy

## Methods with proposed solutions

The first approach of feature extractor is to use ResNet-based fine-tuning, i.e., extracting the first few layers of ResNet34 and ResNet50 as a feature extractor, which respectively uses several groups of basic block(ResNet34-based) and several groups of bottleneck block(ResNet50-based).

The second approach of feature extractor is built by using an automatic hyper-parameter optimization framework called optuna, which can dynamically construct the search spaces for the hyper-parameters. For a given search space of hyper-parameters, optuna will go through different trials to find the several optimal combination of hyperparameters. Based on the optimal combination of parameters found by optuna and then fine-tuned artificially.

## The Model Overview and Results
![Architecture_Overview](https://github.com/ChenErdi/MasterArbeit/blob/main/IMG/Architecture_Overview.JPG)
![Architecture_Overview02](https://github.com/ChenErdi/MasterArbeit/blob/main/IMG/Architecture_Overview02.png)

---

![dm_optuna_v1_1stripe](https://github.com/ChenErdi/MasterArbeit/blob/main/IMG/dm_optuna_v1_3stripes_01.JPG)
