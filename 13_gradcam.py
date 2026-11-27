"""Generate a Grad-CAM overlay for a MobileNetV2 prediction."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
BASE=Path(__file__).resolve().parent
MODEL=BASE/"models"/"mobilenetv2_waste_model.keras"
OUT=BASE/"results"/"gradcam"
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]

def create_heatmap(model,batch,class_id=None):
    backbone=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model) and "mobilenetv2" in layer.name.lower())
    conv=backbone.get_layer("out_relu")
    grad_model=tf.keras.Model(model.inputs,[conv.output,model.output])
    with tf.GradientTape() as tape:
        activations,probs=grad_model(batch,training=False)
        if class_id is None: class_id=tf.argmax(probs[0])
        loss=probs[:,class_id]
    grads=tape.gradient(loss,activations)
    weights=tf.reduce_mean(grads,axis=(1,2))
    heat=tf.reduce_sum(activations[0]*weights[0][None,None,:],axis=-1)
    heat=tf.maximum(heat,0); heat/=tf.reduce_max(heat)+1e-8
    return heat.numpy(),probs.numpy()[0],int(class_id.numpy())

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("image"); args=parser.parse_args()
    path=Path(args.image)
    if not path.is_file(): raise FileNotFoundError(path)
    if not MODEL.is_file(): raise FileNotFoundError("Train MobileNetV2 first.")
    original=cv2.imread(str(path))
    if original is None: raise ValueError("Could not read image.")
    rgb=cv2.cvtColor(cv2.resize(original,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)
    model=tf.keras.models.load_model(MODEL)
    heat,probs,index=create_heatmap(model,tf.convert_to_tensor(rgb[None,...]))
    heat=cv2.resize(heat,(original.shape[1],original.shape[0]))
    colored=cv2.applyColorMap(np.uint8(255*heat),cv2.COLORMAP_JET)
    overlay=cv2.addWeighted(original,.58,colored,.42,0)
    OUT.mkdir(parents=True,exist_ok=True); target=OUT/f"{path.stem}_gradcam.jpg"
    cv2.imwrite(str(target),overlay)
    print("Prediction:",CLASSES[index],f"{probs[index]*100:.2f}%"); print("Grad-CAM:",target)

if __name__=="__main__": main()
