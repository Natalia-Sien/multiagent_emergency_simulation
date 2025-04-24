import tkinter as tk
from multiprocessing import Value
import time

"""
A very simple progress gui to just show when we saved and at what stage we are at
"""
def progress_gui(progress: Value, total: int):
    #create main window
    root = tk.Tk()
    #set window title
    root.title("Training Progress")

    #label showing numeric progress
    label = tk.Label(root, text="0 / " + str(total), font=("Arial", 16))
    label.pack(padx=20, pady=20)

    #canvas for drawing progress bar background
    progressbar = tk.Canvas(root, width=300, height=25, bg="#eee", bd=0, highlightthickness=0)
    #rectangle item for the fill bar, starts at zero width
    bar = progressbar.create_rectangle(0, 0, 0, 25, fill="#4caf50")
    progressbar.pack(pady=10)

    #function to update GUI elements periodically
    def update():
        #read shared progress value safely
        with progress.get_lock():
            current = progress.value
        #update label text with formatted numbers
        label.config(text=f"{current:,} / {total:,}")
        #calculate new fill width proportional to progress
        fill_width = int((current / total) * 300)
        #resize the rectangle to new width
        progressbar.coords(bar, 0, 0, fill_width, 25)

        #debug print to console
        print(f"Progress: {current} / {total}")

        #schedule next update if not complete
        if current < total:
            root.after(200, update)
        else:
            #display completion message when done
            label.config(text="Training Complete!")

    #start the periodic updates
    update()
    #enter the Tkinter main event loop
    root.mainloop()