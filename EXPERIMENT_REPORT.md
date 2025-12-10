# RNN vs MBRA Experiment Report

**Date**: December 10, 2025  
**Experimenter**: Joy Li  
**Project**: Agar.io AI - Comparing Genetic Algorithms for RNNs and Model-Based Reflex Agents

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| RNN Agents | 5 |
| MBRA Agents | 5 |
| Number of Simulations | 10 |
| Map Size | 1500 x 1500 |
| Food Count | 600 |
| Virus Count | 20 |
| Duration per Simulation | 300 seconds (5 minutes) |
| Total Frames per Simulation | 18000 frames (60 FPS) |

---

## Results

### Overall Statistics

| Metric | RNN Agents | MBRA Agents |
|--------|-----------|-------------|
| Sample Size | 50 (5 x 10) | 50 (5 x 10) |
| Mean Fitness | -305.13 | 172.61 |
| Standard Deviation | 299.51 | 354.55 |
| Minimum | -481.81 | -478.96 |
| Maximum | 380.08 | 891.99 |

### Per-Simulation Results

| Simulation | RNN Alive | MBRA Alive | RNN Avg Fitness | MBRA Avg Fitness |
|------------|-----------|------------|-----------------|------------------|
| 1 | 2/5 | 3/5 | -219.75 | 44.41 |
| 2 | 1/5 | 4/5 | -296.11 | 243.91 |
| 3 | 1/5 | 4/5 | -333.05 | 241.86 |
| 4 | 1/5 | 5/5 | -321.69 | 278.91 |
| 5 | 1/5 | 3/5 | -311.46 | 183.03 |
| 6 | 1/5 | 3/5 | -326.63 | 228.78 |
| 7 | 1/5 | 4/5 | -319.27 | 135.88 |
| 8 | 1/5 | 5/5 | -305.89 | 299.99 |
| 9 | 1/5 | 3/5 | -298.82 | 36.04 |
| 10 | 1/5 | 3/5 | -318.59 | 33.25 |

### Survival Rate

| Agent Type | Total Survived | Survival Rate |
|------------|----------------|---------------|
| RNN | 11/50 | 22% |
| MBRA | 37/50 | 74% |

---

## Analysis

### 1. MBRA Significantly Outperforms RNN

- MBRA mean fitness (172.61) is significantly higher than RNN mean fitness (-305.13)
- Difference: 477.74 points
- MBRA agents consistently survive longer and collect more food

### 2. Causes of Negative RNN Fitness

The negative fitness values for RNN agents are primarily due to:
- Death penalty (death_weight = 500.0): Most RNN agents died during simulations
- Short survival time: RNN agents failed to effectively avoid threats

### 3. MBRA Advantages

- Memory buffer system: Can remember positions of objects that left the field of view
- Priority decay mechanism: Effectively distinguishes importance of threats, prey, and food
- Rule-driven decision making: Can quickly make escape decisions when near threats

### 4. Variance Analysis

- RNN standard deviation = 299.51: Moderate variance with mostly negative fitness
- MBRA standard deviation = 354.55: Similar variance but with positive mean
  - Best case: 891.99 (successfully collected food and survived)
  - Worst case: -478.96 (died in some simulations)

---

## Text for Paper Abstract

```
The RNNs achieved an average fitness score of -305.13 (sigma=299.51) 
compared to the MBRAs average fitness of 172.61 (sigma=354.55).
```

Extended version:

```
In 10 simulations with 5 agents of each type on a 1500x1500 map with 
600 food pellets and 20 viruses, the MBRAs significantly outperformed 
the RNNs. The RNNs achieved an average fitness score of -305.13 
(sigma=299.51), compared to the MBRAs average fitness of 172.61 
(sigma=354.55). The MBRA agents demonstrated superior performance 
with an improvement of 477.74 points over the RNN agents.
```

---

## Fitness Calculation Formula

```
fitness = (food_weight * food_eaten +
           time_alive_weight * time_alive_ratio +
           cells_eaten_weight * cells_eaten +
           score_weight * score
           - death_weight * died)
```

Parameter values:
- food_weight = 0.5
- time_alive_weight = 100.0
- cells_eaten_weight = 50.0
- score_weight = 0.75
- death_weight = 500.0

---

## Conclusions

1. MBRA significantly outperforms RNN in this experiment
2. RNN requires more training generations or better hyperparameter tuning
3. The memory buffer system in MBRA is highly effective in dynamic environments
4. Future work: Increase RNN training generations, adjust fitness weights

---

Report generated: December 10, 2025 (Updated with fixed MBRA agent)
