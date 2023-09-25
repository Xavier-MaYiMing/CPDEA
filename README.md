### CPDEA: Convergence-penalized density evolutionary algorithm

##### Reference: Liu Y, Ishibuchi H, Yen G G, et al. Handling imbalance between convergence and diversity in the decision space in evolutionary multimodal multiobjective optimization[J]. IEEE Transactions on Evolutionary Computation, 2019, 24(3): 551-565.

##### CPDEA is a multi-objective evolutionary algorithm (MOEA) to solve multi-modal multi-objective optimization problem (MMOPs) with imbalance between convergence and diversity in the decision space.

| Variables | Meaning                                              |
| --------- | ---------------------------------------------------- |
| npop      | Population size                                      |
| iter      | Iteration number                                     |
| lb        | Lower bound                                          |
| ub        | Upper bound                                          |
| K         | The K-nearest parameter (default = 3)                |
| eta       | Local convergence parameter (default = 2)            |
| eta_c     | Spread factor distribution index (default = 20)      |
| eta_m     | Perturbance factor distribution index (default = 20) |
| nvar      | The dimension of decision space                      |
| pop       | Population                                           |
| objs      | Objectives                                           |
| arch      | Archive                                              |
| arch_objs | The objectives of archive                            |
| fDKN      | Double K-nearest neighbor value                      |
| D_Dec     | Distance in decision space                           |
| dom       | Domination matrix                                    |
| fCPD      | Convergence-penalied density                         |

#### Test problem: IDMP-M2-T1

$$
\begin{aligned}
	& f_1(x) = \min(|x_1 + 0.6| + g_1(x), |x_1 - 0.4| + g_2(x)) \\
	& f_2(x) = \min(|x_1 + 0.4| + g_1(x), |x_1 - 0.6| + g_2(x)) \\
	& g_1(x) = |x_2 + 0.5| \\
	& g_2(x) = \alpha |x_2 - 0.5|, \quad \alpha \geq 1 \\
	& x_1 \in [-1, 1], \ x_2 \in [-1, 1]
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(60, 18000, np.array([-1, -1]), np.array([1, 1]))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/CPDEA/blob/main/Pareto%20front.png)

![](https://github.com/Xavier-MaYiMing/CPDEA/blob/main/Pareto%20set.png)



