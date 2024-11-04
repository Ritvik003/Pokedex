import tkinter as tk
from tkinter import filedialog, Label, Button, Radiobutton, Frame, StringVar, Canvas, Scrollbar, Toplevel
from PIL import Image, ImageTk
import os
import extract
import numpy as np

def ssd(v1, v2):
    return np.sum((v1 - v2)**2)

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_similar_images(input_image_filename, similarity_measure, feature_type):
    feats_df = extract.get_features(feature_type)  # Assuming get_features returns a DataFrame
    input_image_filename = os.path.basename(input_image_filename)
    input_features = feats_df.loc[feats_df['Image'] == input_image_filename, 'Features'].values[0]

    if similarity_measure == "ssd":
        feats_df['Similarity'] = feats_df['Features'].apply(lambda x: ssd(input_features, x))
    elif similarity_measure == "cosine":
        feats_df['Similarity'] = feats_df['Features'].apply(lambda x: cosine(input_features, x))

    similar_images_df = feats_df.sort_values(by='Similarity', ascending=(similarity_measure == "ssd"))
    similar_images_df = similar_images_df[similar_images_df['Image'] != input_image_filename]

    return similar_images_df.head(10)['Image']

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png")])
    return file_path

def show_images(filenames, db_path):
    top = Toplevel()
    top.title("Similar Images")
    canvas = Canvas(top)
    scrollbar = Scrollbar(top, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    frame = Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor='nw')

    for fname in filenames:
        img_path = os.path.join(db_path, fname)
        img = Image.open(img_path)
        img.thumbnail((150, 150))
        img = ImageTk.PhotoImage(img)
        label = Label(frame, image=img)
        label.image = img
        label.pack(padx=5, pady=5)

    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

def main_window():
    root = tk.Tk()
    root.title("Image Similarity Viewer")

    banner_label = Label(root, text="Pokemon, Land of Fun!", font=('Helvetica', 16, 'bold'))
    banner_label.pack(side=tk.TOP, fill=tk.X)

    logo = Image.open("/Users/ritvikwarrier/Desktop/HW8-FinalProject/pokemon.jpg")
    logopic = ImageTk.PhotoImage(logo)
    label = Label(root, image=logopic)
    label.image = logopic
    label.pack(side=tk.TOP, anchor=tk.W)

    similarity = StringVar(value="ssd")
    feat_type = StringVar(value="avg_color")

    Radiobutton(root, text="SSD", variable=similarity, value="ssd").pack(anchor=tk.W)
    Radiobutton(root, text="Cosine", variable=similarity, value="cosine").pack(anchor=tk.W)

    Radiobutton(root, text="Average Color", variable=feat_type, value="AVG_COLOR").pack(anchor=tk.W)
    Radiobutton(root, text="Spatial", variable=feat_type, value="SPATIAL").pack(anchor=tk.W)
    Radiobutton(root, text="Histogram", variable=feat_type, value="HIST").pack(anchor=tk.W)
    Radiobutton(root, text="HoG", variable=feat_type, value="HOG").pack(anchor=tk.W)

    # Button to upload image and show similar images
    Button(root, text="Upload and Find Similar Images", command=lambda: show_images(
        get_similar_images(upload_image(), similarity.get(),feat_type.get()),
        "/Users/ritvikwarrier/Desktop/HW8-FinalProject/Data/Database")).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main_window()
