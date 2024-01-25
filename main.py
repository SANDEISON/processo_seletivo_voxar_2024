# Biblioteca para c
import streamlit as st
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import torch

resnet = models.resnet101(pretrained=True)


preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

def valida_imagem(img):
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    resnet.eval()
    out = resnet(batch_t)
    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    posicao = int(index[0])

    nome = labels[index[0]], percentage[index[0]].item()
    st.write(nome)

def verifica_animal_imagem(imagens):
    data = []
    for uploaded_file in imagens:
        imagem = Image.open(uploaded_file)
        valida_imagem(imagem)
        data.append(uploaded_file.name)
    df = pd.DataFrame(data=data)
    return df

#Título da Página
st.write("## Reconhecimento Animais com PyTorch")
#Criei uma variavel uploaded_files para receber uma lista de arquivos de imagens, onde só possivel selecionar arquvios do tipo indicado, com limite de 200 MB
uploaded_files = st.file_uploader("Escolha suas imagens ", type=['png', 'jpg', 'JPEG'] ,accept_multiple_files=True)

if uploaded_files:
    btn = st.button("Reconhecer Animais")
    if btn:
        csv = verifica_animal_imagem(uploaded_files)
        st.write(csv)
        btn2 = st.download_button(
            label="Download arquivo cvs",
            data=csv.to_csv().encode('utf-8'),
            file_name="Animais.csv",
            mime='text/csv',
        )

