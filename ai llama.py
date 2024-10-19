import tkinter as tk
from tkinter import scrolledtext
import ollama
import threading

class OllamaChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("Ollama Chat Interface")
        master.geometry("600x400")

        self.chat_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=70, height=20)
        self.chat_display.pack(padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        self.input_field = tk.Entry(master, width=50)
        self.input_field.pack(side=tk.LEFT, padx=10)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=5)

        self.input_field.bind("<Return>", lambda event: self.send_message())

        self.desired_model = 'llama3.2:latest'

    def send_message(self):
        user_message = self.input_field.get()
        if user_message:
            self.display_message("You: " + user_message)
            self.input_field.delete(0, tk.END)
            threading.Thread(target=self.get_ai_response, args=(user_message,)).start()

    def get_ai_response(self, user_message):
        response = ollama.chat(model=self.desired_model, messages=[
            {'role': 'user', 'content': user_message},
        ])
        ai_response = response['message']['content']
        self.display_message("AI: " + ai_response)

    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    gui = OllamaChatGUI(root)
    root.mainloop()