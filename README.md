# Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning

This repository contains code and resources for training and evaluating adversarial agents against victim models in a controlled environment. The following instructions will guide you through setting up the environment, training victim models, training adversarial agents, and evaluating their performance.

---

## Table of Contents
1. [Setting Up Docker and Dependencies](#setting-up-docker-and-dependencies)
2. [Training Victim Models](#training-victim-models)
3. [Training Adversarial Agents](#training-adversarial-agents)
4. [Evaluating Adversarial Agents and Benchmarks](#evaluating-adversarial-agents-and-benchmarks)

---

## 1. Setting Up Docker and Dependencies 

To ensure a consistent and reproducible environment, we use Docker. Follow these steps to set up the environment:

1. **Install Docker**:
    - Download and install Docker from [https://www.docker.com/](https://www.docker.com/).
    - Verify the installation:
      ```bash
      docker --version
      ```

2. **Build the Docker Image**:
    - Navigate to the project directory:
      ```bash
      cd ./adversarial-agents
      ```
    - Build the Docker image:
      ```bash
      docker build -t adversarial-agents .
      ```

3. **Run the Docker Container**:
    - Start a container with the built image:
      ```bash
      docker run -it --rm -v $(pwd):/workspace adversarial-agents
      ```

4. **Install Additional Dependencies** (if needed):
    - Inside the container, install Python dependencies:
      ```bash
      pip install -r requirements.txt
      ```

---

## 2. Training the Victim Model

Victim models are the targets for adversarial attacks. Follow these steps to train and save them for experimentation:

1. **Train the ResNet50 Victim Model**:
    - Run the training script:
      ```bash
      python train_victim_model.py --dataset cifar10 --epochs 50
      ```
    - Adjust the parameters (`--dataset`, `--epochs`, `--lr`, `--batch-size`) as needed for your experiments.

2. **Save the Model**:
    - The trained model will automatically be saved in the `victim_models/` directory with a filename like `<dataset>+resnet50.pth`

---

## 3. Training Adversarial Agents

Adversarial agents are trained using reinforcement learning (RL) to improve subsequent attacks on victim models. Follow these steps to train your adversarial agents:

1. **Train the Adversarial Agent**:
    - Run the training script:
      ```bash
      python train_agent.py --dataset cifar10 --rl_attack_type RLMaxLoss --epsilon 0.35 --N 5 --theta 0.05 --training_steps 5000000 --label rlmaxloss_agent
      ```
    
    - Adjust the parameters (`--dataset`, `--rl_attack_type`, `--epsilon`, `--N`, `--theta`, `--training_steps`, `--label`) as needed for your experiments.

    - The `--rl_attack_type` can be set to `RLMaxLoss` or `RLMinNorm` depending on the attack strategy you want to implement (described in 'Learning Adversarial Policies' section of the paper).

2. **Monitor Training**:
    - Training logs will be saved in the `learning_results/` directory. You can monitor the training progress by checking these logs.

    - `learning_results/episodic.csv`
        - Contains the episodic rewards, distortion, queries, original label $y$, and prediction $Z(x_t)$
        - PPO updates happen every 4096 queries, which can be added as a column post-training by adding a running sum of the queries and floor dividing by 4096.
    - `learning_results/progress.csv`
        - Contains default training logs from StableBaselines3 for RL metrics. For more information, refer to the logger output section of the StableBaselines3 documentation [here](https://stable-baselines3.readthedocs.io/en/master/common/logger.html#explanation-of-logger-output).

3. **Save the Agent**:
    - The trained adversarial agent will be saved in the `agents/` directory with a filename like `<label>.pth`.
    - For example, if you set `--label rlmaxloss_agent`, the agent will be saved as `agents/rlmaxloss_agent.pth`.

---

## 4. Evaluating Adversarial Agents and Benchmarks

Evaluate the performance of adversarial agents against victim models and compare with benchmarks:

1. **Run Adversarial Agent Evaluation**:
    - Execute the evaluation script:
      ```bash
      python evaluate_agent.py --dataset cifar10 --rl_attack_type RLMaxLoss --epsilon 0.35 --N 5 --theta 0.05 --agent_file ./agents/rlmaxloss_agent.pth
      ```
    - **Note**: Attack parameters (`--rl_attack_type`, `--epsilon`, `--N`, `--theta`) should match those used during training for consistency.

2. **Analyze Adversarial Agents Results**:
    - The evaluation logs will be saved in the `evaluation_results/` directory.
    - `evaluation_results/episodic.csv`
        - Contains the episodic rewards, distortion, queries, original label $y$, and prediction $Z(x_t)$.

3. **Run Square Attack Evaluation**:
    - Execute the square attack evaluation script:
      ```bash
      python evaluate_square.py --dataset cifar10 --epsilon 0.35 --n_queries 1000
      ```
    - **Note**: Results will be output in the standard output detailing hyperparameters, attack success rate, query budget, and other metrics.

---
