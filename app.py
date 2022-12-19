# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os

model = load_model('keras_model.h5')
class_names = open('labels.txt', 'r').readlines()

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("AI画像認識アプリ")
st.sidebar.write("Teachable Machineの学習モデルを使って画像判定します。")

st.sidebar.write("")

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        image = Image.open(img_file)
        size = (224, 224)
        # 画像をセンタリングし指定したsizeに切り出す処理
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        st.image(image, caption="対象の画像", width=480)
        st.write("")
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # 例外処理
        try:
          data[0] = normalized_image_array
        except Exception as e:
          st.write(e)
        else:
          prediction = model.predict(data)

        # 円グラフの表示
        pie_labels = class_names
        pie_probs = prediction[0]
        st.subheader('円グラフ')
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)
        # 一覧表の表示
        st.subheader('一覧表')
        st.write(pd.DataFrame(pie_probs,pie_labels))
