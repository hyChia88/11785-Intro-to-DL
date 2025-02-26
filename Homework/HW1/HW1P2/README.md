# 11785 HW1P2 Submission
huiyenc, Chia Hui Yen, 15/2/2025
## Log:
For my model submission, I experimented with various architectures and hyperparameters to optimize performance. Initially, I tested different learning rates, batch sizes, activation functions, and optimizers. My first attempts is using SDG, and i read from forum said AdamW is better, and the few early attempts (1–2) used smaller batch sizes, AdamW, and GELU activation. 

Later that I found the modal training is work, I later switched to ReLU with AdamW (Attempts 3–6), incorporating batch normalization and dropout to improve generalization, I gradually increase batchsize within the range that didnt run that much time (30mins per epochs etc).

Then, me and my teammate, tried diffirent network structure and I primarily focused on a pyramid network structure, gradually increasing its depth and complexity. In later attempts (5–6), I expanded the network width (increasing layer sizes from 2048 to 4096 neurons) and added more layers. I also explored architectural modifications, such as transitioning from a pyramid to a cylindrical structure with varied dropout rates.

As I iterated, my model accuracy improved, with Attempt 3 achieving ~81.63% accuracy and Attempt 5_2 reaching 82.22%. However, later adjustments (6_1, 6_2) had mixed results, showing that deeper and more complex architectures didn’t always lead to better performance, and even starting decay after 20+ epochs. So the final submission stops at 82.22%, pyramid network structure with 20 epochs.

## Important details of model:
- wandb link: https://wandb.ai/hychia2024-carnegie-mellon-university/hw1p2/runs/2o4jgus4?nw=nwuserhychia2024