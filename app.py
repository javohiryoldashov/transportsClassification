import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#title
st.title('Three type of vhicl classification')

file = st.file_uploader('Rasm yuklash', type=['png', 'jpg', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('transport_model.pkl')

    pred, pred_id, probs = model.predict(img)

    st.success(pred)
    st.info(f'Extimolik: {probs[pred_id]*100:.1f}')

    fig = px.bar(x = probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)