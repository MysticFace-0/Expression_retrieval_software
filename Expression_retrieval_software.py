import tkinter as tk
import math
import tkinter.filedialog
from tkinter import messagebox

from PIL import Image,ImageTk
import pandas as pd
from tkinter.messagebox import showinfo

from PyTorch.FER_image import FER_image
from PyTorch.FER_live_cam import FER_live_cam

root = tk.Tk()
root.title('Facial Expression Retrieval Software')
root.geometry('800x550+100+100')  # 界面大小

# root.attributes("-alpha",0.9)#界面半透明
root["background"]='#FFFFFF'

font_song = ('宋体', 15)
font_bold = ('黑体', 15)

canvas = tk.Canvas(root, width = 800, height = 800)
canvas.pack()
img = ImageTk.PhotoImage(Image.open("./interface_picture/all.png"))
canvas.create_image(400, 250, image=img)

"""点击事件"""

def click_button_Picture(self=None):  # KHcoder模板输出
    filename = tk.filedialog.askopenfilename()
    if filename =='':
        pass
    FER_image(filename)


def click_button_Video(self=None):  # 空白文体模板输出
    FER_live_cam()

def click_button_Help(self=None):  # 空白文体模板输出
    a=messagebox.askquestion('Help',"Welcome!!!\n"+"Picture: Upload a picture to recognize its expression\n"+
    "Video: Use local cameras to identify the faces of the\n"+"people in front of the camera",type= "ok")

button_Picture = tk.Button(root, text='Picture', width=10, font=font_bold, relief=tk.GROOVE, bg='#ffffff',
                      command=click_button_Picture).place(x=150, y=500)

button_Video = tk.Button(root, text='Video', width=10, font=font_bold, relief=tk.GROOVE, bg='#ffffff',
                           command=click_button_Video).place(x=350, y=500)

button_help = tk.Button(root, text='Help', width=10, font=font_bold, relief=tk.GROOVE, bg='#ffffff',
                           command=click_button_Help).place(x=550, y=500)

root.mainloop()
