"""Train a custom CNN using the shared train/validation/test folders."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,
    f1_score,precision_score,recall_score,ConfusionMatrixDisplay)
BASE=Path(__file__).resolve().parent; DATA=BASE/"shared_dataset"; MODELS=BASE/"models"; RESULTS=BASE/"results"
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]; SIZE=(224,224); BATCH=32; SEED=42

def load_split(split,shuffle):
    return tf.keras.utils.image_dataset_from_directory(DATA/split,class_names=CLASSES,
      image_size=SIZE,batch_size=BATCH,shuffle=shuffle,seed=SEED).prefetch(tf.data.AUTOTUNE)

def build_model():
    inputs=tf.keras.Input(shape=(*SIZE,3)); x=tf.keras.layers.Rescaling(1/255)(inputs)
    for filters in (32,64,128):
        x=tf.keras.layers.Conv2D(filters,3,padding="same",activation="relu")(x)
        x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(.35)(x)
    outputs=tf.keras.layers.Dense(len(CLASSES),activation="softmax")(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

def main():
    if not (DATA/"manifest.csv").is_file(): raise FileNotFoundError("Run 00_prepare_shared_dataset.py first.")
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    train,val,test=(load_split("train",True),load_split("val",False),load_split("test",False))
    model=build_model()
    counts=[sum(1 for p in (DATA/"train"/name).rglob("*") if p.is_file()) for name in CLASSES]
    if any(count==0 for count in counts): raise ValueError("Every class must have training images.")
    total=sum(counts)
    class_weights={i: total/(len(CLASSES)*count) for i,count in enumerate(counts)}
    print("Training class weights:",class_weights)
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=3,min_lr=1e-6),
      tf.keras.callbacks.ModelCheckpoint(MODELS/"best_cnn_model.keras",monitor="val_loss",save_best_only=True)]
    history=model.fit(train,validation_data=val,epochs=30,callbacks=callbacks,class_weight=class_weights)
    best=tf.keras.models.load_model(MODELS/"best_cnn_model.keras")
    y=np.concatenate([labels.numpy() for _,labels in test]); probs=best.predict(test,verbose=1); pred=probs.argmax(1)
    cm=confusion_matrix(y,pred,labels=list(range(6)))
    metrics={"model":"Custom CNN","evaluation":"test","split_id":"shared_dataset_seed_42_70_15_15",
      "test_count":int(len(y)),"classes":CLASSES,"accuracy":float(accuracy_score(y,pred)),
      "macro_precision":float(precision_score(y,pred,average="macro",zero_division=0)),
      "macro_recall":float(recall_score(y,pred,average="macro",zero_division=0)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "weighted_f1":float(f1_score(y,pred,average="weighted",zero_division=0)),
      "classification_report":classification_report(y,pred,labels=list(range(6)),target_names=CLASSES,output_dict=True,zero_division=0),
      "confusion_matrix":cm.tolist()}
    best.save(MODELS/"cnn_waste_model.keras")
    (RESULTS/"cnn_results.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(xticks_rotation=35)
    plt.tight_layout(); plt.savefig(RESULTS/"cnn_confusion_matrix.png",dpi=220); plt.close()
    for key,values in history.history.items():
        plt.figure(); plt.plot(values,label=key); plt.title("CNN "+key); plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS/f"cnn_{key}.png",dpi=180); plt.close()
    print("Custom CNN test macro-F1:",metrics["macro_f1"])

if __name__=="__main__": main()
