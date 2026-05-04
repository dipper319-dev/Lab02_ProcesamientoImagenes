import cv2
import numpy as np
import joblib
import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from skimage.feature import hog, local_binary_pattern

# ── Rutas dinámicas ─────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")

# ── Cargar modelos ──────────────────────────────────────
svm_hog    = joblib.load(os.path.join(MODELOS_DIR, "svm_HOG.pkl"))
svm_lbp    = joblib.load(os.path.join(MODELOS_DIR, "svm_LBP.pkl"))
scaler_hog = joblib.load(os.path.join(MODELOS_DIR, "scaler_HOG.pkl"))
scaler_lbp = joblib.load(os.path.join(MODELOS_DIR, "scaler_LBP.pkl"))

CLASES  = {0: "🐱 Gato", 1: "🐶 Perro", 2: "🌍 Otro"}
COLORES = {0: "#a6e3a1", 1: "#89b4fa", 2: "#f9e2af"}

# ── Funciones de extracción ─────────────────────────────
def preprocesar(ruta):
    img = cv2.imread(ruta)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def extraer_hog(img):
    feat, _ = hog(img, orientations=9, pixels_per_cell=(8,8),
                  cells_per_block=(2,2), visualize=True)
    return feat

def extraer_lbp(img):
    lbp = local_binary_pattern(img, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist

def clasificar(ruta):
    img = preprocesar(ruta)

    f_hog    = scaler_hog.transform([extraer_hog(img)])
    pred_hog = svm_hog.predict(f_hog)[0]
    prob_hog = svm_hog.predict_proba(f_hog)[0][pred_hog]

    f_lbp    = scaler_lbp.transform([extraer_lbp(img)])
    pred_lbp = svm_lbp.predict(f_lbp)[0]
    prob_lbp = svm_lbp.predict_proba(f_lbp)[0][pred_lbp]

    return pred_hog, prob_hog, pred_lbp, prob_lbp

# ── Interfaz ────────────────────────────────────────────
class App:
    def __init__(self, root):
        root.title("🐾 Clasificador Gato vs Perro vs Otro")
        root.geometry("520x650")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        Label(root, text="🐾 Clasificador de Animales",
              font=("Helvetica", 18, "bold"),
              bg="#1e1e2e", fg="#cdd6f4").pack(pady=15)

        self.frame_img = Frame(root, bg="#313244",
                               width=300, height=300,
                               relief="ridge", bd=2)
        self.frame_img.pack(pady=5)
        self.frame_img.pack_propagate(False)

        self.lbl_img = Label(self.frame_img, bg="#313244",
                             text="📂 Carga una imagen",
                             fg="#6c7086", font=("Helvetica", 12))
        self.lbl_img.pack(expand=True)

        Button(root, text="📂  Cargar Imagen",
               command=self.cargar,
               font=("Helvetica", 12, "bold"),
               bg="#89b4fa", fg="#1e1e2e",
               relief="flat", padx=20, pady=8,
               cursor="hand2").pack(pady=15)

        frame_res = Frame(root, bg="#1e1e2e")
        frame_res.pack(pady=5)

        Label(frame_res, text="HOG",
              font=("Helvetica", 11, "bold"),
              bg="#1e1e2e", fg="#a6e3a1").grid(row=0, column=0, padx=40)
        Label(frame_res, text="LBP",
              font=("Helvetica", 11, "bold"),
              bg="#1e1e2e", fg="#f38ba8").grid(row=0, column=1, padx=40)

        self.lbl_hog = Label(frame_res, text="—",
                             font=("Helvetica", 18, "bold"),
                             bg="#1e1e2e", fg="#a6e3a1")
        self.lbl_hog.grid(row=1, column=0, padx=40, pady=5)

        self.lbl_lbp = Label(frame_res, text="—",
                             font=("Helvetica", 18, "bold"),
                             bg="#1e1e2e", fg="#f38ba8")
        self.lbl_lbp.grid(row=1, column=1, padx=40, pady=5)

        self.lbl_prob_hog = Label(frame_res, text="",
                                  font=("Helvetica", 10),
                                  bg="#1e1e2e", fg="#6c7086")
        self.lbl_prob_hog.grid(row=2, column=0)

        self.lbl_prob_lbp = Label(frame_res, text="",
                                  font=("Helvetica", 10),
                                  bg="#1e1e2e", fg="#6c7086")
        self.lbl_prob_lbp.grid(row=2, column=1)

        Label(root, text="Decisión final:",
              font=("Helvetica", 11),
              bg="#1e1e2e", fg="#6c7086").pack(pady=(20, 0))

        self.lbl_decision = Label(root, text="—",
                                  font=("Helvetica", 24, "bold"),
                                  bg="#1e1e2e", fg="#f9e2af")
        self.lbl_decision.pack()

        self.lbl_confianza = Label(root, text="",
                                   font=("Helvetica", 10),
                                   bg="#1e1e2e", fg="#6c7086")
        self.lbl_confianza.pack()

    def cargar(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not ruta:
            return

        img_pil = Image.open(ruta).resize((300, 300))
        img_tk  = ImageTk.PhotoImage(img_pil)
        self.lbl_img.configure(image=img_tk, text="")
        self.lbl_img.image = img_tk

        pred_hog, prob_hog, pred_lbp, prob_lbp = clasificar(ruta)

        self.lbl_hog.config(text=CLASES[pred_hog], fg=COLORES[pred_hog])
        self.lbl_lbp.config(text=CLASES[pred_lbp], fg=COLORES[pred_lbp])
        self.lbl_prob_hog.config(text=f"Confianza: {prob_hog*100:.1f}%")
        self.lbl_prob_lbp.config(text=f"Confianza: {prob_lbp*100:.1f}%")

        votos    = [pred_hog, pred_lbp]
        decision = max(set(votos), key=votos.count)
        self.lbl_decision.config(text=CLASES[decision], fg=COLORES[decision])

        if pred_hog == pred_lbp:
            conf_final = (prob_hog + prob_lbp) / 2
            self.lbl_confianza.config(text=f"Confianza promedio: {conf_final*100:.1f}%")
        else:
            self.lbl_confianza.config(text="⚠️ Los descriptores no coinciden")

# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()