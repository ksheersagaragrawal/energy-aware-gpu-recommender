# Energy-Efficient GPU Recommendation

This project explores ML-based GPU power modeling for energy-efficient GPU recommendation.

The main idea is simple:

Given a game requirement and a list of candidate GPUs, we want to recommend a GPU that is strong enough to run the game, but does not waste unnecessary power.

## Project Direction

We use two datasets:

1. **Game Requirements Dataset**
   - Gives minimum and recommended GPU requirements for games.
   - Used to understand what level of GPU capability a game needs.

2. **TechPowerUp GPU Specs Dataset**
   - Gives real GPU specifications such as memory, bandwidth, clocks, TMUs, ROPs, TDP, and suggested PSU.
   - Used as the candidate GPU database.
   - Also used to train power prediction models.

## Current Scope

We are focusing on **desktop discrete GPUs**.

We are not using mobile GPUs, integrated GPUs, data center GPUs, or workstation GPUs in the main experiment because they follow different power, memory, and deployment assumptions.

## Game Dataset Cleaning

For the game requirement dataset:

- Removed CPU columns because this project is focused on GPU recommendation.
- Ignored columns that were mostly missing, redundant, or not useful for GPU selection.
- Split the dataset into:
  - minimum requirement dataset
  - recommended requirement dataset
- Kept around 16 useful GPU-related columns.
- Categorized columns into:
  - hard filters
  - soft filters
  - performance score features
  - optional metadata
  - not-needed columns

## Requirement Vector

The requirement vector represents what the game needs.

Example fields:

- GPU memory / VRAM
- DirectX
- texture rate
- pixel rate
- memory bandwidth
- TMUs
- ROPs

These are used to filter candidate GPUs.

## Performance Score

After filtering, we need a way to rank GPUs.

We create a performance score using performance-related GPU specs such as:

- texture rate
- pixel rate
- memory bandwidth
- TMUs
- ROPs
- memory speed
- boost clock

We do not include columns like DirectX, PSU, power connector, OS, RAM, or HDD in the performance score because they are not direct GPU performance features.

## Power Modeling

The power model will be trained on the TechPowerUp GPU specs dataset.

Primary target:

- `Board Design__TDP`

Secondary target:

- `Board Design__Suggested PSU`

TDP is closer to actual GPU board power, while suggested PSU is more of a system-level provisioning metric.

## Recommendation Flow

The final flow is:

1. Clean the game requirement dataset.
2. Clean the TechPowerUp GPU specs dataset.
3. Train a model to predict GPU TDP / PSU.
4. Create a requirement vector for each game.
5. Filter GPUs that do not satisfy the game requirement.
6. Compute performance score for remaining GPUs.
7. Rank feasible GPUs by performance-per-watt.
8. Recommend the best GPU.

## Baselines

We plan to compare our method against these baselines:

1. **Minimum GPU baseline**
   - Find the GPU closest to the game’s minimum requirement profile.

2. **Recommended GPU baseline**
   - Find the GPU closest to the game’s recommended requirement profile.

3. **Lowest TDP feasible**
   - Among feasible GPUs, choose the one with the lowest TDP.

4. **Lowest PSU feasible**
   - Among feasible GPUs, choose the one with the lowest suggested PSU.

5. **Performance-per-watt**
   - Among feasible GPUs, choose the GPU with the best performance score per watt.

## Evaluation Metrics

We will evaluate each method using:

- average selected TDP
- average selected PSU
- average performance score
- average performance-per-watt
- performance over-provisioning
- power over-provisioning
- efficiency regret

## Important Notes

This project does not claim to predict exact FPS or runtime gaming latency.

The performance score is a static hardware-based proxy.

The goal is to build a simple and interpretable energy-aware GPU recommender using game requirements and GPU specifications.