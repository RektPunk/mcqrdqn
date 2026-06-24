<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=MCQRDQN&height=120&fontSize=90">
</div>

This project provides a robust implementation of **Monotone Conposite Quantile Regression DQN (MCQR-DQN)** in PyTorch. Standard Distributional RL methods, such as QR-DQN, often suffer from **quantile crossing**—a phenomenon where the predicted quantiles lose their order, resulting in physically impossible or unstable value distributions. This implementation addresses this issue by employing a monotonicity constraint via a [l1-penalizing method (l1pm)](https://github.com/RektPunk/l1pm).

## Visualization
*Trained agents (Seed 42, Episode 0) performing across various environments.*

<div align="center">
  
| MountainCar-v0 (Episode 0) | LunarLander-v3 (Episode 0) |
| :---: | :---: |
| <img width="400" src="https://github.com/user-attachments/assets/4d839f41-f813-4844-a70e-062634bb8b65" /> |<img width="400" src="https://github.com/user-attachments/assets/f5a124d2-4239-4bd5-b139-1716c6fa42f7" /> |


| Acrobot-v1 (Episode 0) | CartPole-v1 (Episode 0) |
| :---: | :---: |
| <img width="400" src="https://github.com/user-attachments/assets/99882ac2-6bc2-4286-928a-a8e25c8a5f55" /> | <img width="400" height="320" src="https://github.com/user-attachments/assets/38513ee0-b1c3-45ca-bad9-485ee206f00e" /> |

</div>
