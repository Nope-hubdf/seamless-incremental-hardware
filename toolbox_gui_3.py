import tkinter as tk
from tkinter import ttk
import random, string
from datetime import date

# ---------------- FUNCTIONS ---------------- #
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return "Error: Division by zero!" if b == 0 else a / b

def celsius_to_fahrenheit(c): return (c * 9/5) + 32
def fahrenheit_to_celsius(f): return (f - 32) * 5/9

def generate_password(length=8):
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(chars) for _ in range(length))

def roll_dice(): return random.randint(1, 6)
def calculate_total(prices): return sum(prices)
def calculate_age(birth_year): return date.today().year - birth_year
def word_count(text): return len(text.split())


# ---------------- MAIN WINDOW ---------------- #
root = tk.Tk()
root.title("🎒 Python Beginner Toolbox")
root.geometry("600x500")

style = ttk.Style()
style.theme_use("clam")  # base theme

# ---------------- THEME SWITCHER ---------------- #
dark_mode = False

def toggle_theme():
    global dark_mode
    dark_mode = not dark_mode
    if dark_mode:
        root.config(bg="#1e1e1e")
        style.configure(".", background="#1e1e1e", foreground="white", fieldbackground="#333")
        style.configure("TLabel", background="#1e1e1e", foreground="white")
        style.configure("TEntry", fieldbackground="#333", foreground="white")
        style.configure("TButton", background="#444", foreground="white")
        theme_btn.config(text="☀️ Light Mode")
    else:
        root.config(bg="#f4f4f9")
        style.configure(".", background="#f4f4f9", foreground="black", fieldbackground="white")
        style.configure("TLabel", background="#f4f4f9", foreground="black")
        style.configure("TEntry", fieldbackground="white", foreground="black")
        style.configure("TButton", background="#eee", foreground="black")
        theme_btn.config(text="🌙 Dark Mode")

theme_btn = ttk.Button(root, text="🌙 Dark Mode", command=toggle_theme)
theme_btn.pack(pady=5)

# Notebook
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

def make_entry(parent, width=30):
    entry = ttk.Entry(parent, font=("Segoe UI", 11), width=width)
    entry.pack(pady=5)
    return entry

def make_result_label(parent):
    lbl = ttk.Label(parent, text="", font=("Segoe UI", 11, "bold"))
    lbl.pack(pady=10)
    return lbl


# ---------------- TAB 1: CALCULATOR ---------------- #
calc_tab = ttk.Frame(notebook)
notebook.add(calc_tab, text="🧮 Calculator")

ttk.Label(calc_tab, text="Enter first number:").pack(pady=5)
calc_entry1 = make_entry(calc_tab)
ttk.Label(calc_tab, text="Enter second number:").pack(pady=5)
calc_entry2 = make_entry(calc_tab)
calc_result = make_result_label(calc_tab)

def do_calculator():
    try:
        a = float(calc_entry1.get())
        b = float(calc_entry2.get())
        result = (f"Add: {add(a,b):.2F} | "
                  f"Subtract: {subtract(a,b):.2F} | "
                  f"Multiply: {multiply(a,b):.2F} | "
                  f"Divide: {divide(a,b):.2F}")
        calc_result.config(text=result)
    except ValueError:
        calc_result.config(text="❌ Please enter valid numbers")

ttk.Button(calc_tab, text="Calculate", command=do_calculator).pack(pady=10)


# ---------------- TAB 2: TEMPERATURE ---------------- #
temp_tab = ttk.Frame(notebook)
notebook.add(temp_tab, text="🌡 Temperature")

ttk.Label(temp_tab, text="Enter Celsius:").pack(pady=5)
celsius_entry = make_entry(temp_tab)
ttk.Label(temp_tab, text="Enter Fahrenheit:").pack(pady=5)
fahrenheit_entry = make_entry(temp_tab)
temp_result = make_result_label(temp_tab)

def do_temp():
    try:
        c = float(celsius_entry.get())
        f = float(fahrenheit_entry.get())
        result = (f"{c}°C = {celsius_to_fahrenheit(c):.2f}°F | "
                  f"{f}°F = {fahrenheit_to_celsius(f):.2f}°C")
        temp_result.config(text=result)
    except ValueError:
        temp_result.config(text="❌ Enter valid numbers")

ttk.Button(temp_tab, text="Convert", command=do_temp).pack(pady=10)


# ---------------- TAB 3: PASSWORD ---------------- #
pass_tab = ttk.Frame(notebook)
notebook.add(pass_tab, text="🔑 Password")

ttk.Label(pass_tab, text="Enter password length:").pack(pady=5)
pass_entry = make_entry(pass_tab)
pass_result = make_result_label(pass_tab)

def do_password():
    try:
        length = int(pass_entry.get())
        pass_result.config(text=generate_password(length))
    except ValueError:
        pass_result.config(text="❌ Enter a valid number")

ttk.Button(pass_tab, text="Generate", command=do_password).pack(pady=10)


# ---------------- TAB 4: DICE ---------------- #
dice_tab = ttk.Frame(notebook)
notebook.add(dice_tab, text="🎲 Dice Roller")

dice_result = make_result_label(dice_tab)

def do_dice():
    dice_result.config(text=f"You rolled a {roll_dice()} 🎲")

ttk.Button(dice_tab, text="Roll Dice", command=do_dice).pack(pady=20)


# ---------------- TAB 5: SHOPPING ---------------- #
shop_tab = ttk.Frame(notebook)
notebook.add(shop_tab, text="🛒 Shopping")

ttk.Label(shop_tab, text="Enter prices (space separated):").pack(pady=5)
shop_entry = make_entry(shop_tab)
shop_result = make_result_label(shop_tab)

def do_total():
    try:
        prices = [float(x) for x in shop_entry.get().split()]
        shop_result.config(text=f"Total: {calculate_total(prices):.2f}")
    except ValueError:
        shop_result.config(text="❌ Enter valid prices")

ttk.Button(shop_tab, text="Calculate Total", command=do_total).pack(pady=10)


# ---------------- TAB 6: AGE ---------------- #
age_tab = ttk.Frame(notebook)
notebook.add(age_tab, text="🎂 Age Calculator")

ttk.Label(age_tab, text="Enter your birth year:").pack(pady=5)
age_entry = make_entry(age_tab)
age_result = make_result_label(age_tab)

def do_age():
    try:
        year = int(age_entry.get())
        age_result.config(text=f"You are {calculate_age(year)} years old 🎉")
    except ValueError:
        age_result.config(text="❌ Enter a valid year")

ttk.Button(age_tab, text="Calculate Age", command=do_age).pack(pady=10)


# ---------------- TAB 7: WORD COUNTER ---------------- #
word_tab = ttk.Frame(notebook)
notebook.add(word_tab, text="📝 Word Counter")

ttk.Label(word_tab, text="Enter text:").pack(pady=5)
word_entry = make_entry(word_tab, width=40)
word_result = make_result_label(word_tab)

def do_wordcount():
    text = word_entry.get()
    word_result.config(text=f"Word count: {word_count(text)}")

ttk.Button(word_tab, text="Count Words", command=do_wordcount).pack(pady=10)


# ---------------- RUN APP ---------------- #
toggle_theme()  # start in dark mode by default
root.mainloop()