## <center>🐍贪吃蛇游戏及基于强化学习的模型🤖</center>

#### English version README included. 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **从 0 到 22000 分的进化！** 本项目展示了一个基于 DQN 的智能体如何通过迁移学习，从掌握基础生存到玩转复杂的 Combo & Frenzy 机制。你可以自己上手游玩，也可以训练或观看Agent游玩


## ⚡ Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Play the Game

```bash
python eating_snake.py
```

### 3️⃣ Watch Trained AI Play

```bash
python agent_test.py
```


---

## 🚀 核心亮点 (Key Highlights)

### 1. 迁移学习策略 (Transfer Learning Strategy)
这是本项目成功的关键。由于带有 Combo 机制的环境奖励函数极其复杂，直接训练容易导致模型陷入局部最优。
* **阶段一：基础课** - 在经典贪吃蛇环境下训练 1000 局，建立基础避障和寻路逻辑。
* **阶段二：专业课** - 迁移权重至“加强版”环境。AI 带着“先验知识”进场，仅用 100 局即实现分数爆发。

### 2. 精炼的 11 维状态特征 (11-D State Representation)
放弃了沉重的像素输入，采用了高效的 11 维向量，确保了模型在 CPU 上也能极速收敛：
- **危险探测** (3维)：前方、左侧、右侧是否有障碍。
- **移动方向** (4维)：当前前进方向的 One-hot 编码。
- **食物定位** (4维)：食物相对于蛇头的上下左右位置。

### 3. 自定义游戏机制 (Advanced Mechanics)
不同于传统贪吃蛇，本项目在 Pygame 环境中引入了：
- **Combo 系统**：短时间内连续吃豆分数成倍增长。
- **Frenzy Mode**：触发狂暴模式，获得超高额得分奖励。

### 4. 开箱即玩 (Play Now!)
你可以自己上手游玩，也可以直接用与训练好的模型进行游玩。


---

## 📊 训练表现 (Performance)

* **最高得分 (Max Score):** `22000+`
* **收敛速度:** 迁移学习后，模型迅速适应，能够稳定的到达1000+分
* **稳定性:** 在 $Epsilon = 0.05$ 的探索率下，依然能保持极高的高分触发频率。



---

## 📂 文件结构 (Project Structure)

```text
├── agent_test.py # 测试贪吃蛇Agent       
├── agent_training.py # 训练贪吃蛇Agent
├── classic_agent_test.py # 测试经典贪吃蛇Agent
├── classic_agent_training.py # 训练经典贪吃蛇Agent
├── eating_snake.py # 可以自己上手玩的贪吃蛇
├── requirements.txt  # 环境依赖
├── snake_agent.pth # 贪吃蛇Agent模型
├── snake_memory.pkl # 贪吃蛇Agent记忆
├── classic_snake_agent.pth # 经典贪吃蛇Agent模型
└── classic_snake_memory.pkl # 经典贪吃蛇Agent记忆
```

***


## <center>🐍 Snake Game with Reinforcement Learning Agent 🤖</center>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **From 0 to 22,000+ score evolution!**  
> This project demonstrates how a DQN-based agent learns progressively through transfer learning — from basic survival skills to mastering complex Combo & Frenzy mechanics.  
> You can play the game yourself, train the agent, or watch the trained AI in action.

---

## 🚀 Key Highlights

### 1. Transfer Learning Strategy

This is the core reason behind the success of the project.  
Due to the highly complex reward structure introduced by the Combo system, direct training often leads to local optima.

* **Stage 1: Fundamentals**  
  Train the agent for 1000 episodes in the classic Snake environment to learn collision avoidance and pathfinding.

* **Stage 2: Advanced Training**  
  Transfer the learned weights into the enhanced environment.  
  With prior knowledge, the agent achieves score explosion within only 100 episodes.

---

### 2. Efficient 11-D State Representation

Instead of computationally expensive pixel inputs, we use an efficient 11-dimensional feature vector, allowing fast convergence even on CPU:

- **Danger Detection (3 dims)**  
  Whether there is danger straight, left, or right.

- **Movement Direction (4 dims)**  
  One-hot encoding of the current movement direction.

- **Food Location (4 dims)**  
  Relative position of the food (up, down, left, right).

---

### 3. Advanced Game Mechanics

Unlike traditional Snake, this project introduces several custom mechanics implemented in Pygame:

- **Combo System**  
  Consecutive food consumption within a short time window multiplies rewards.

- **Frenzy Mode**  
  Special state with significantly boosted scoring potential.

---

### 4. Play Now!

You can:

- 🎮 Play manually
- 🤖 Watch the trained AI play
- 🧠 Train the agent yourself

---

## 📊 Performance

- **Max Score:** `22000+`
- **Convergence Speed:** After transfer learning, the agent rapidly adapts and consistently reaches 1000+ scores.
- **Stability:** Even with an exploration rate of $Epsilon = 0.05$, the agent maintains a high frequency of high-score runs.

---

## 📂 Project Structure

```text
├── agent_test.py                 # Test enhanced Snake agent
├── agent_training.py             # Train enhanced Snake agent
├── classic_agent_test.py         # Test classic Snake agent
├── classic_agent_training.py     # Train classic Snake agent
├── eating_snake.py               # Playable Snake game
├── requirements.txt              # Dependencies
├── snake_agent.pth               # Trained enhanced model
├── snake_memory.pkl              # Replay memory
├── classic_snake_agent.pth       # Classic model
└── classic_snake_memory.pkl      # Classic replay memory


```
