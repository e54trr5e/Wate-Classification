"""Streamlit image classifier, Grad-CAM explanation and SQLite prediction analytics."""
import json
import sqlite3
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
BASE=Path(__file__).resolve().parent; MODEL=BASE/"models"/"mobilenetv2_waste_model.keras"
CAL=BASE/"results"/"mobilenetv2_confidence_calibration.json"; DB=BASE/"results"/"predictions.sqlite3"
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]

def db():
    DB.parent.mkdir(exist_ok=True); con=sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
      filename TEXT NOT NULL, predicted_class TEXT NOT NULL,
      confidence REAL NOT NULL, requires_review INTEGER NOT NULL)""")
    con.commit(); return con

@st.cache_resource
def load_model(): return tf.keras.models.load_model(MODEL)

def explain(model,tensor,class_id):
    base=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model) and "mobilenetv2" in layer.name.lower())
    grad_model=tf.keras.Model(model.inputs,[base.get_layer("out_relu").output,model.output])
    with tf.GradientTape() as tape:
        conv,probs=grad_model(tensor,training=False); loss=probs[:,class_id]
    grads=tape.gradient(loss,conv); weights=tf.reduce_mean(grads,axis=(1,2))
    heat=tf.reduce_sum(conv[0]*weights[0][None,None,:],axis=-1)
    heat=tf.maximum(heat,0); heat/=tf.reduce_max(heat)+1e-8
    return heat.numpy()

st.set_page_config(page_title="Smart Waste Classifier",page_icon="♻️",layout="wide")
st.title("♻️ AI Smart Waste Classification")
st.caption("MobileNetV2 · Grad-CAM · validation-calibrated review warning · SQLite analytics")
if not MODEL.is_file(): st.error("Model missing. Run 12_train_mobilenetv2.py first."); st.stop()
threshold=.70
if CAL.is_file(): threshold=float(json.loads(CAL.read_text()).get("threshold",threshold))
uploaded=st.file_uploader("Upload a waste image",type=["jpg","jpeg","png","webp"])
if uploaded:
    pil=Image.open(uploaded).convert("RGB"); rgb=np.asarray(pil)
    tensor=tf.convert_to_tensor(tf.image.resize(rgb,(224,224))[None,...],dtype=tf.float32)
    model=load_model(); probs=model.predict(tensor,verbose=0)[0]
    index=int(np.argmax(probs)); confidence=float(probs[index]); label=CLASSES[index]
    review=confidence<threshold
    a,b=st.columns(2)
    with a: st.image(pil,caption="Input image",use_container_width=True)
    with b:
        st.subheader("Prediction: "+label.title()); st.metric("Confidence",f"{confidence*100:.1f}%")
        if review: st.warning(f"Below validation threshold ({threshold*100:.1f}%). Manual verification recommended.")
        else: st.success("Prediction passed the validation-calibrated review threshold.")
        st.write("Class probabilities"); st.bar_chart({c:float(probs[i]) for i,c in enumerate(CLASSES)})
    if st.checkbox("Show Grad-CAM explanation"):
        heat=explain(model,tensor,index); heat=cv2.resize(heat,(rgb.shape[1],rgb.shape[0]))
        colored=cv2.applyColorMap(np.uint8(255*heat),cv2.COLORMAP_JET)
        overlay=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.58,colored,.42,0)
        st.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB),caption="Grad-CAM heatmap overlay",use_container_width=True)
    if st.button("Save prediction to history",type="primary"):
        with db() as con:
            con.execute("INSERT INTO predictions(created_at,filename,predicted_class,confidence,requires_review) VALUES(?,?,?,?,?)",
              (datetime.now(timezone.utc).isoformat(),uploaded.name,label,confidence,int(review)))
        st.success("Prediction saved to SQLite.")

st.divider(); st.subheader("Prediction analytics")
with db() as con:
    counts=con.execute("SELECT predicted_class,COUNT(*) FROM predictions GROUP BY predicted_class ORDER BY predicted_class").fetchall()
    total=con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    recent=con.execute("SELECT created_at,filename,predicted_class,confidence,requires_review FROM predictions ORDER BY id DESC LIMIT 25").fetchall()
x,y=st.columns(2); x.metric("Saved predictions",total); y.metric("Categories recorded",len(counts))
if counts: st.bar_chart({name:count for name,count in counts})
if recent:
    st.dataframe([{"Time (UTC)":t,"Image":f,"Prediction":p,"Confidence %":round(c*100,1),"Manual review":bool(r)}
      for t,f,p,c,r in recent],use_container_width=True)
