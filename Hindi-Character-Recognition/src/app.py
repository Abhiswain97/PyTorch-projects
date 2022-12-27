import streamlit as st
import torch
import config as CFG
from PIL import Image
import numpy as np
import json
from model import HNet

def classify(model, file, mapping):
    image = Image.open(file).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).unsqueeze(0)

    outputs = model(image)

    _, preds = torch.max(outputs, 1)

    st.markdown(
        f"<h2>The predicted character is: {mapping[str(preds[0].item())]}</h2>",
        unsafe_allow_html=True,
    )

st.markdown("<h1>Hindi Character Recognition<h1>", unsafe_allow_html=True)

option = st.sidebar.radio(label="Classify Hindi Digit or Vyanjan ?", options=["Digit", "Vyanjan"], index=0)

def upload_and_classify(model, mapping):
    file = st.file_uploader("Upload image!")

    if file is not None:

        st.image(file, use_column_width=True)
        button = st.button("Predict")

        if button:
            classify(model=model, file=file, mapping=mapping)

mapping, model = None, None

if option == "Digit":
    if CFG.BEST_MODEL_DIGIT.exists():
        model = HNet(num_classses=10)
        model.load_state_dict(torch.load(CFG.BEST_MODEL_DIGIT))
        with open("index_to_digit.json", "r") as f:
            mapping = json.load(f)
        upload_and_classify(model, mapping)
    else:
        st.error("No model exists! First Train model!")
else:
    if CFG.BEST_MODEL_VYANJAN.exists():
        model = HNet(num_classses=36)
        model.load_state_dict(torch.load(CFG.BEST_MODEL_VYANJAN))
        with open("index_to_vyanjan.json", "r") as f:
            mapping = json.load(f)
        upload_and_classify(model, mapping)
    else:
        st.error("No model exists! First train model!")
