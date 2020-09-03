import tkinter as tk
from tkinter import *

class Application(tk.Frame):
    def __init__(self, master=None, names = ["Angelina" , "Chris", "Jennifer", "Elle"]):
        super().__init__(master)

        self.master = master
        self.pack()
        self.create_widgets()
        listbox = Listbox(self)
        listbox.pack()
        listbox.bind('<ButtonRelease-1>', self.get_list)

        for item in names:
            listbox.insert(END, item)

        self.listbox = listbox  # make listbox a class property so it
        # can be accessed anywhere within the class

    def create_widgets(self):
        self.OK= tk.Button(self)
        self.OK["text"] = "OK"
        self.OK["command"] = self.OK
        self.OK.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def Ok(self):
        index = self.listbox.curselection()[0]
        self.seltext = self.listbox.get(index)


    def get_list(self, event):
            index = self.listbox.curselection()[0]
            self.seltext = self.listbox.get(index)



