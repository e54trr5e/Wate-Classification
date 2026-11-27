"""Train MobileNetV2 transfer learning on the exact shared dataset split."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,
    f1_score,precision_score,recall_score,ConfusionMatrixDisplay)
BASE=Path(__file__).resolve().parent; DATA=BASE/"shared_dataset"; MODELS=BASE/"models"; RESULTS=BASE/"results"
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]; SIZE=(224,224); BATCH=32; SEED=42

def dataset(split,shuffle):
    return tf.keras.utils.image_dataset_from_directory(DATA/split,class_names=CLASSES,
      image_size=SIZE,batch_size=BATCH,shuffle=shuffle,seed=SEED).prefetch(tf.data.AUTOTUNE)

def main():
    if not (DATA/"manifest.csv").is_file(): raise FileNotFoundError("Run 00_prepare_shared_dataset.py first.")
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    train,val,test=(dataset("train",True),dataset("val",False),dataset("test",False))
    augmentation=tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal"),tf.keras.layers.RandomRotation(.08),
      tf.keras.layers.RandomZoom(.10),tf.keras.layers.RandomContrast(.10)],name="augmentation")
    backbone=tf.keras.applications.MobileNetV2(input_shape=(*SIZE,3),include_top=False,weights="imagenet")
    backbone.trainable=False
    inputs=tf.keras.Input(shape=(*SIZE,3)); x=augmentation(inputs)
    x=tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x=backbone(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(.30)(x); outputs=tf.keras.layers.Dense(6,activation="softmax")(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    counts=[sum(1 for p in (DATA/"train"/name).rglob("*") if p.is_file()) for name in CLASSES]
    if any(count==0 for count in counts): raise ValueError("Every class must have training images.")
    total=sum(counts)
    class_weights={i: total/(len(CLASSES)*count) for i,count in enumerate(counts)}
    print("Training class weights:",class_weights)
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=2,min_lr=1e-6),
      tf.keras.callbacks.ModelCheckpoint(MODELS/"mobilenetv2_head_best.keras",monitor="val_loss",save_best_only=True)]
    print("Stage 1: train classification head with frozen ImageNet backbone.")
    h1=model.fit(train,validation_data=val,epochs=12,callbacks=callbacks,class_weight=class_weights)
    model=tf.keras.models.load_model(MODELS/"mobilenetv2_head_best.keras")
    backbone=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model) and "mobilenetv2" in layer.name.lower())
    backbone.trainable=True
    for layer in backbone.layers[:-30]: layer.trainable=False
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=False
    callbacks[2]=tf.keras.callbacks.ModelCheckpoint(MODELS/"mobilenetv2_finetuned_best.keras",monitor="val_loss",save_best_only=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    print("Stage 2: fine-tune upper 30 backbone layers with low learning rate.")
    h2=model.fit(train,validation_data=val,epochs=15,callbacks=callbacks,class_weight=class_weights)
    head_best=tf.keras.models.load_model(MODELS/"mobilenetv2_head_best.keras")
    fine_best=tf.keras.models.load_model(MODELS/"mobilenetv2_finetuned_best.keras")
    head_val=float(head_best.evaluate(val,verbose=0)[0])
    fine_val=float(fine_best.evaluate(val,verbose=0)[0])
    best=head_best if head_val<=fine_val else fine_best
    print("Validation loss — frozen head:",head_val,"fine-tuned:",fine_val)
    best.save(MODELS/"mobilenetv2_waste_model.keras")
    y=np.concatenate([labels.numpy() for _,labels in test]); probs=best.predict(test,verbose=1); pred=probs.argmax(1)
    cm=confusion_matrix(y,pred,labels=list(range(6)))
    report={"model":"MobileNetV2 Transfer Learning","evaluation":"test",
      "split_id":"shared_dataset_seed_42_70_15_15","test_count":int(len(y)),"classes":CLASSES,
      "backbone":"MobileNetV2 ImageNet","fine_tuned_layers":30,
      "accuracy":float(accuracy_score(y,pred)),
      "macro_precision":float(precision_score(y,pred,average="macro",zero_division=0)),
      "macro_recall":float(recall_score(y,pred,average="macro",zero_division=0)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "weighted_f1":float(f1_score(y,pred,average="weighted",zero_division=0)),
      "classification_report":classification_report(y,pred,labels=list(range(6)),target_names=CLASSES,output_dict=True,zero_division=0),
      "confusion_matrix":cm.tolist()}
    (RESULTS/"mobilenetv2_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(xticks_rotation=35)
    plt.tight_layout(); plt.savefig(RESULTS/"mobilenetv2_confusion_matrix.png",dpi=220); plt.close()
    for key,values in h2.history.items():
        plt.figure(); plt.plot(values,label=key); plt.title("MobileNetV2 fine-tuning "+key); plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS/f"mobilenetv2_{key}.png",dpi=180); plt.close()
    print("MobileNetV2 test macro-F1:",report["macro_f1"])

if __name__=="__main__": main()
