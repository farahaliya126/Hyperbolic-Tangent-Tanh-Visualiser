import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Tanh Visualiser", page_icon="📈", layout="wide")
st.title("📈 Hyperbolic Tangent (Tanh)")
st.latex(r"\tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}\qquad \tanh'(x)=1-\tanh^{2}(x)")


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


# ---- Sidebar controls ----
st.sidebar.header("Controls")
x_min, x_max = st.sidebar.slider("x-axis range", -10.0, 10.0, (-6.0, 6.0), 0.5)
x_in = st.sidebar.slider("Input value x", float(x_min), float(x_max), 1.0, 0.1)
show_grad = st.sidebar.checkbox("Show derivative", True)

x = np.linspace(x_min, x_max, 400)
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, tanh(x), label="Tanh(x)", color="tab:purple", lw=2)
    if show_grad:
        ax.plot(x, tanh_grad(x), "--", label="Derivative", color="tab:orange")
    ax.scatter([x_in], [tanh(x_in)], color="red", zorder=5, label="Your input")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("x (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.metric("Tanh(x)", f"{tanh(x_in):.4f}")
    st.metric("Derivative", f"{tanh_grad(x_in):.4f}")
    st.markdown(
        "**Key points**\n"
        "- Output range: (-1, 1), zero-centred\n"
        "- Steeper than sigmoid (max gradient 1)\n"
        "- Still saturates for large |x| (vanishing gradients)\n"
        "- Common in RNN/LSTM hidden states"
    )
