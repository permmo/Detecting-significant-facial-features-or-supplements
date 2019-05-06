from tkinter import filedialog
from tkinter import *
from tkinter import ttk
from threaded_task import ThreadedTask

def browse_button():
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)


def browse_button_2():
    global folder_path2
    filename = filedialog.askdirectory()
    folder_path2.set(filename)


def start_button_press(images_folder, save_folder, selected_model):
    if(images_folder.get()== ''):
        erro_text = StringVar()
        erro_text.set("Nespravne zadana cesta k priecinku")
        error_label = Label(master=root, textvariable=erro_text,fg="red")
        error_label.place(x=290, y=420)
        return
    if (save_folder.get() == ''):
        erro_text = StringVar()
        erro_text.set("Nespravne zadana cesta k priecinku")
        error_label = Label(master=root, textvariable=erro_text,fg="red")
        error_label.place(x=290, y=420)
        return
    if (selected_model.get() == 0):
        erro_text = StringVar()
        erro_text.set("Vyberte model")
        error_label = Label(master=root, textvariable=erro_text,fg="red")
        error_label.place(x=340, y=420)
        return

    working_text = StringVar()
    working_text.set("                             ...Spracuvavam...                             ")
    working_label = Label(master=root, textvariable=working_text)
    working_label.place(x=220, y=420)
    state_text = StringVar()
    state_text.set("0/50")
    state_label = Label(master=root, textvariable=state_text)
    state_label.place(x=375, y=480)

    task = ThreadedTask(
        progressbar,
        state_text,
        images_folder,
        save_folder,
        selected_model.get(),
        working_text,
        sort_var.get(),
        hightlight_var.get())
    task.start()


def sort_checkbutton_press():
    if(sort_var.get() == 1):
        hightlight_checkbutton.place(x=400, y=270)
    else:
        hightlight_checkbutton.place_forget()


root = Tk()
root.geometry("800x550+300+300")
root.title("MatusApp")

folder_path = StringVar()
folder_path2 = StringVar()
radio_button_var = IntVar()
sort_var=IntVar()
hightlight_var = IntVar()
first_label_text = StringVar()
second_label_text = StringVar()
third_label_text = StringVar()

start_button = Button(text="Start", width=15, bg ='#00FF80',
                      command= lambda: start_button_press(folder_path, folder_path2, radio_button_var))
start_button.place(x=590, y=310)

sort_checkbutton = Checkbutton(text="Roztriedit podla charakteristik/doplnkov",
                               variable=sort_var, command=sort_checkbutton_press)
sort_checkbutton.place(x=50, y=270)

hightlight_checkbutton = Checkbutton(text="Zvyraznit charakteristiky/doplnky",
                                     variable=hightlight_var)

first_label_text.set("Vyberte priecinok z obrazkami.")
first_label= Label(master=root, textvariable=first_label_text, fg="blue")
first_label.place(x=50, y=30)

path1_label= Label(master=root, textvariable=folder_path)
path1_label.place(x=220, y=52)

browse1_button = Button(text="Vybrat...", height=1, width=15, command=browse_button)
browse1_button.place(x=50, y=50)

second_label_text.set("Vyberte priecinok kam sa maju obrazky ulozit.")
second_label= Label(master=root, textvariable=second_label_text, fg="blue")
second_label.place(x=50, y=100)

path2_label = Label(master=root, textvariable=folder_path2)
path2_label.place(x=220, y=122)

browse2_button = Button(text="Vybrat...", height=1, width=15, command=browse_button_2)
browse2_button.place(x=50, y=120)

third_label_text.set("Vyberte model.")
third_label= Label(master=root, textvariable=third_label_text, fg="blue")
third_label.place(x=50, y=170)

radiobutton1 = Radiobutton(root, text="faster_rcnn_inception_v2_coco",
                           variable=radio_button_var, value=1)
radiobutton1.place(x=50, y=190)

radiobutton2 = Radiobutton(root, text="faster_rcnn_resnet101_coco",
                           variable=radio_button_var, value=2)
radiobutton2.place(x=400, y=190)

radiobutton3 = Radiobutton(root, text="faster_rcnn_inception_resnet_v2_atrous_coco",
                           variable=radio_button_var, value=3)
radiobutton3.place(x=400, y=210)

radiobutton4 = Radiobutton(root, text="faster_rcnn_resnet50_coco",
                           variable=radio_button_var, value=4)
radiobutton4.place(x=50, y=210)

progressbar = ttk.Progressbar(root, orient="horizontal", length=700)

mainloop()