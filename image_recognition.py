import customtkinter as ctk
import cv2
from PIL import Image
import numpy as np

class FaceDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CodeAlpha AI Face Detector")
        self.geometry("600x750")

        self.label = ctk.CTkLabel(self, text="AI Face Recognition Tool", font=("Arial", 22, "bold"))
        self.label.pack(pady=20)

        self.btn_select = ctk.CTkButton(self, text="Upload Image 📸", command=self.upload_image)
        self.btn_select.pack(pady=10)

        self.img_display = ctk.CTkLabel(self, text="No Image Selected")
        self.img_display.pack(pady=20)

        self.result_label = ctk.CTkLabel(self, text="Status: Ready", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = ctk.filedialog.askopenfilename()
        if file_path:
            try:
                # الحل السحري للمسارات العربية: قراءة الملف كـ Binary أولاً
                with open(file_path, "rb") as f:
                    chunk = f.read()
                
                # تحويل البيانات لمصفوفة NumPy
                nparr = np.frombuffer(chunk, np.uint8)
                # فك تشفير الصورة (Decoding)
                image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image_cv is None:
                    self.result_label.configure(text="Error: Could not decode image!")
                    return

                # تحويل الصورة لرمادي تمهيداً لاكتشاف الوجوه
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

                # تحميل مبرد الوجوه
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                # اكتشاف الوجوه
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50))

                # رسم المربعات
                for (x, y, w, h) in faces:
                    cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # تحويلها لـ RGB للعرض في Tkinter
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(image_rgb)
                
                # عرض الصورة
                display_img = ctk.CTkImage(light_image=img_pil, size=(400, 400))
                self.img_display.configure(image=display_img, text="")
                self.result_label.configure(text=f"Detected: {len(faces)} face(s)")

            except Exception as e:
                self.result_label.configure(text=f"Error: {str(e)}")

if __name__ == "__main__":
    app = FaceDetectorApp()
    app.mainloop()