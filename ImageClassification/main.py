import tkinter as tk
import customtkinter as ctk 
from PIL import ImageTk, Image
from authtoken import auth_token
import torch
from diffusers import StableDiffusionPipeline

# Creating app
app = tk.Tk() 
app.geometry("532x622")
app.title('Stable Bud')
ctk.set_appearance_mode("dark")         

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512, text="")  # Empty text removes 'CTkLabel'
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    text_prompt = prompt.get()
    image = pipe(text_prompt, guidance_scale=8.5).images[0]  # Use `.images[0]`
    
    img = ImageTk.PhotoImage(image)
    lmain.image = img  # Store reference to prevent garbage collection
    lmain.configure(image=img)

    image.save('generatedimage.png')

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", text="Generate", command=generate)
trigger.place(x=206, y=60)

app.mainloop()