import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# ===== ORION Vision Imports =====
from src.model import BayesianCNN
from src.uncertainity import mc_dropout_predict
from src.agent import OrionAgent

# ===== ORION Chat Imports =====
from src.text_uncertainty import text_uncertainty
from src.agent import OrionChatAgent   # make sure this exists

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== App Title =====
st.set_page_config(page_title="ORION Agent", layout="wide")
st.title("🧠 ORION — Uncertainty-Aware Agentic AI")

# ===== Sidebar =====
mode = st.sidebar.selectbox(
    "Select ORION Mode",
    ["Vision Defense Agent", "Chat Agent"]
)

# =========================================================
# 🛡️ VISION MODE — Adversarial Defense Agent
# =========================================================
if mode == "Vision Defense Agent":
    st.header("🛡️ Vision-Based Adversarial Defense")

    # Load model
    model = BayesianCNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    agent = OrionAgent(model)

    uploaded = st.file_uploader(
        "Upload a handwritten digit (MNIST-like)",
        type=["png", "jpg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("L")
        st.image(image, caption="Input Image", width=200)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0).to(device)

        pred, uncertainty = mc_dropout_predict(model, tensor)
        decision = agent.decide(pred, uncertainty)

        st.subheader("🧠 ORION Decision")
        st.write(f"**Prediction:** {pred}")
        st.write(f"**Uncertainty:** {uncertainty:.4f}")
        st.write(f"**Decision:** {decision['decision']}")
        st.write(f"**Reason:** {decision['reason']}")

# =========================================================
# 🤖 CHAT MODE — Uncertainty-Aware Chat Agent
# =========================================================
elif mode == "Chat Agent":
    st.header("🤖 ORION Chat — Self-Aware AI Assistant")

    chat_agent = OrionChatAgent()

    query = st.text_input("Ask ORION something:")

    if query:
        # 🔁 Simulated multiple LLM responses
        # (Replace with real LLM later)
        responses = [
            "This is one possible interpretation of your question.",
            "There could be another explanation depending on context.",
            "I'm not fully confident without additional information."
        ]

        uncertainty = text_uncertainty(responses)
        decision = chat_agent.decide(responses[0], uncertainty)

        st.subheader("🧠 ORION Reasoning")
        st.write(f"**Uncertainty:** {uncertainty:.2f}")
        st.write(f"**Decision:** {decision}")

        if decision == "ANSWER":
            st.success(responses[0])
        elif decision == "ASK_CLARIFICATION":
            st.warning("Can you please clarify your question?")
        else:
            st.error("I’m not confident enough to answer this safely.")

