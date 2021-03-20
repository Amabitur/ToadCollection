from tkinter import *
from PIL import ImageTk, Image
from custom_game import toad_game_custom
from mediapipe_game import toad_game_mediapipe

class App():
    def __init__(self):
        self.root = Tk()
        self.root.geometry('600x400')
        self.root.title('Toad collection')

        img = ImageTk.PhotoImage(Image.open('/home/alena/Downloads/2fbef6cbc36d79fd310d83cb2a897bc4.jpg'))
        panel = Label(self.root, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")

        button = Button(panel, text='Play with custom detector', command=self.start_game_custom, height=5, width=15, wraplength='100p')
        button.pack(side=LEFT)

        button2 = Button(panel, text='Play with mediapipe hands', command=self.start_game_mediapipe, height=5, width=15, wraplength='100p')
        button2.pack(side=RIGHT)

        self.root.mainloop()

    def start_game_custom(self):
        toad_game_custom()

    def start_game_mediapipe(self):
        toad_game_mediapipe()

    def quit(self):
        self.root.destroy()


if __name__ == '__main__':
    App()