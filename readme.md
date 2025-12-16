# ORION — Uncertainty-Aware Agentic AI

ORION is an agentic AI system that reasons about its own uncertainty before
acting. Unlike traditional models that always return an answer, ORION
decides whether to accept, reject, or request clarification based on
predictive uncertainty.

This repository contains:
- A **vision-based adversarial defense agent** using Bayesian Deep Learning
- A **chat-style agent** that estimates uncertainty from response variance
- A unified **agent policy** that drives safe decision-making

---

## 🔍 Core Ideas

- **Bayesian Deep Learning (MC Dropout)** for vision uncertainty
- **Self-consistency / response variance** for text uncertainty
- **Agent-based decision policy** on top of model predictions

---

## 📁 Project Structure

orion/
│
├── src/
│ ├── model.py # Bayesian CNN
│ ├── attacks.py # FGSM attack
│ ├── uncertainty.py # MC Dropout inference
│ ├── text_uncertainty.py # Text uncertainty estimation
│ └── agent.py # ORION decision logic
│
├── streamlit_app.py # Interactive demo (Vision + Chat)
├── model.pth # Trained MNIST model
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
Activate:

Windows

bash
Copy code
venv\Scripts\activate
Linux / macOS

bash
Copy code
source venv/bin/activate
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run ORION Demo
bash
Copy code
streamlit run streamlit_app.py
🧠 ORION Modes
🛡️ Vision Defense Agent
Accepts handwritten digit images (MNIST-like)

Uses Monte Carlo Dropout to estimate uncertainty

Flags high-uncertainty inputs as potentially adversarial

🤖 Chat Agent
Simulates multiple responses to a user query

Estimates uncertainty via response disagreement

Decides to:

Answer

Ask for clarification

Refuse to answer

🚀 Future Extensions
Stronger adversarial attacks (PGD, AutoAttack)

Vision Transformers with Bayesian inference

Real LLM integration (OpenAI / Ollama)

Memory and planning for autonomous agents

Multi-agent ORION systems

📌 Status
Current Version: ORION v0.1
This version focuses on uncertainty-aware perception and decision-making,
forming the core of a larger agentic AI system.
