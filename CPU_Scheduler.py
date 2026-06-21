# =========================================================
# 🔥 AI CPU SCHEDULER WITH ML + RESPONSIVE UI + LIVE SIMULATION
# FCFS | RR | V-RTS-ML | PERFORMANCE GRAPH | COMPARISON TABLE
# MODULE 1 | MODULE 2 | MODULE 4
# =========================================================

import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import matplotlib.pyplot as plt
import time
import copy

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from PIL import Image, ImageTk

# =========================
# PROCESS CLASS
# =========================
class Process:
    def __init__(self, pid, at, bt):
        self.pid = pid
        self.at = at
        self.bt = bt
        self.remaining = bt
        self.ct = 0
        self.wt = 0
        self.tat = 0

# =========================
# ML MODELS
# =========================
Xc = [[2,0],[3,1],[4,2],[6,0],[7,2],[9,3],[12,1],[15,3],[20,0]]
yc = [0,0,0,0,1,1,1,1,1]

clf_model = DecisionTreeClassifier(max_depth=3)
clf_model.fit(Xc, yc)

Xq = [[6,0],[8,1],[10,2],[12,1],[15,3],[20,0]]
yq = [3,4,4,5,6,7]

quantum_model = DecisionTreeRegressor(max_depth=3)
quantum_model.fit(Xq, yq)

def should_split_ml(p):
    return clf_model.predict([[p.bt, p.at]])[0]

def get_ml_quantum(p):
    q = int(round(quantum_model.predict([[p.bt, p.at]])[0]))
    return max(2, min(q, 8))

# =========================
# SCHEDULERS
# =========================
def fcfs(processes):
    time_now = 0
    gantt = []
    plist = sorted(processes, key=lambda x: x.at)

    for p in plist:
        if time_now < p.at:
            time_now = p.at
        gantt.append((p.pid, time_now, time_now + p.bt))
        time_now += p.bt
        p.ct = time_now
        p.tat = p.ct - p.at
        p.wt = p.tat - p.bt
    return gantt

def round_robin(processes, q=4):
    time_now = 0
    gantt = []
    queue = deque()
    plist = sorted(processes, key=lambda x: x.at)
    i = 0

    while queue or i < len(plist):
        while i < len(plist) and plist[i].at <= time_now:
            queue.append(plist[i])
            i += 1

        if not queue:
            time_now += 1
            continue

        p = queue.popleft()
        run = min(q, p.remaining)
        gantt.append((p.pid, time_now, time_now + run))
        time_now += run
        p.remaining -= run

        while i < len(plist) and plist[i].at <= time_now:
            queue.append(plist[i])
            i += 1

        if p.remaining > 0:
            queue.append(p)
        else:
            p.ct = time_now
            p.tat = p.ct - p.at
            p.wt = p.tat - p.bt
    return gantt

def v_rts_ml_scheduler(processes):
    time_now = 0
    gantt = []
    plist = sorted(processes, key=lambda x: x.at)
    completed = 0
    n = len(plist)
    long_rr_queue = deque()

    while completed < n:
        ready = [p for p in plist if p.at <= time_now and p.remaining > 0]

        if not ready:
            time_now += 1
            continue

        short = [p for p in ready if should_split_ml(p) == 0]
        long = [p for p in ready if should_split_ml(p) == 1]

        if short:
            p = min(short, key=lambda x: x.remaining)
            start = time_now
            time_now += 1
            p.remaining -= 1
        else:
            for lp in long:
                if lp not in long_rr_queue:
                    long_rr_queue.append(lp)

            if not long_rr_queue:
                time_now += 1
                continue

            p = long_rr_queue.popleft()
            q = get_ml_quantum(p)
            run = min(q, p.remaining)
            start = time_now
            time_now += run
            p.remaining -= run

            if p.remaining > 0:
                long_rr_queue.append(p)

        if gantt and gantt[-1][0] == p.pid:
            gantt[-1] = (p.pid, gantt[-1][1], time_now)
        else:
            gantt.append((p.pid, start, time_now))

        if p.remaining == 0:
            completed += 1
            p.ct = time_now
            p.tat = p.ct - p.at
            p.wt = p.tat - p.bt
            if p in long_rr_queue:
                long_rr_queue.remove(p)

    return gantt

# =========================
# 📊 GANTT DRAW
# =========================
def draw_gantt(gantt, title):
    plt.close("all")
    plt.figure(figsize=(9, 2))
    for pid, start, end in gantt:
        plt.barh(1, end - start, left=start)
        plt.text((start + end) / 2, 1, pid, ha="center", va="center")
    plt.yticks([])
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# =========================
# PERFORMANCE METRICS
# =========================
def compute_metrics(processes, scheduler):
    plist = copy.deepcopy(processes)
    scheduler(plist)
    avg_wt = sum(p.wt for p in plist) / len(plist)
    avg_tat = sum(p.tat for p in plist) / len(plist)
    return avg_wt, avg_tat

# =========================
# 📈 PERFORMANCE RUNNER (FIXED)
# =========================
def run_performance_analysis():
    p = get_processes()
    if not p:
        return

    results = []
    for name, algo in [
        ("FCFS", fcfs),
        ("RR", round_robin),
        ("V-RTS-ML", v_rts_ml_scheduler)
    ]:
        avg_wt, avg_tat = compute_metrics(p, algo)
        results.append((name, avg_wt, avg_tat))

    methods = [r[0] for r in results]
    avg_wt = [r[1] for r in results]
    avg_tat = [r[2] for r in results]

    x = range(len(methods))
    width = 0.3

    plt.figure(figsize=(8, 4))
    plt.bar([i - width/2 for i in x], avg_wt, width=width, label="Avg WT")
    plt.bar([i + width/2 for i in x], avg_tat, width=width, label="Avg TAT")
    plt.xticks(x, methods)
    plt.ylabel("Time Units")
    plt.title("Average Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# INPUT
# =========================
def get_processes():
    plist = []
    for i, (at, bt) in enumerate(entries):
        if not at.get().isdigit() or not bt.get().isdigit():
            messagebox.showerror("Error", f"P{i+1} Invalid Input")
            return None
        plist.append(Process(f"P{i+1}", int(at.get()), int(bt.get())))
    return plist

# =========================
# 🔥 STABLE SIMULATION (OLD LAYOUT STYLE, NO THREADS)
# =========================
def side_by_side_simulation():
    plist = get_processes()
    if not plist:
        return

    win = tk.Toplevel(root)
    win.title("🧠 Side-by-Side CPU State Simulation (FCFS | RR | ML)")
    canvas = tk.Canvas(win, width=1600, height=720, bg="white")
    canvas.pack()

    # ---------------- BACKGROUND IMAGE ----------------
    bg_img_raw = Image.open(r"C:\SIMATS\capstone\os\cpu-scheduling.webp")
    bg_img_raw = bg_img_raw.resize((1600, 720), Image.LANCZOS)
    bg_img = ImageTk.PhotoImage(bg_img_raw)

    canvas.bg_img = bg_img
    canvas.create_image(0, 0, image=bg_img, anchor="nw", tags="bg")
    canvas.tag_lower("bg")  # Background stays at bottom

    SPEED = 12
    STEPS = 35

    algos = ["FCFS", "RR", "ML"]

    base_x = {
        "FCFS": 350,
        "RR": 800,
        "ML": 1250
    }

    states = ["NEW", "READY", "RUNNING", "BLOCKED", "TERMINATED"]

    ymap = {
        "NEW": 140,
        "READY": 240,
        "RUNNING": 340,
        "BLOCKED": 440,
        "TERMINATED": 540
    }

    BOX_HALF_WIDTH = 190
    BOX_HALF_HEIGHT = 30

    TOKEN_SPACING_X = 75
    TOKENS_PER_ROW = 4

    tokens = {a: {} for a in algos}
    # ---------------- BLACK TAG LABEL ----------------
    def draw_tag(x, y, text, font=("Segoe UI", 11, "bold"),pad_x=10, pad_y=4,bg="black", fg="white", tags="layout"):
        # Draw text first (temporary, to measure size)
        temp = canvas.create_text(x, y, text=text, font=font, fill=fg, tags=tags)
        bbox = canvas.bbox(temp)
        # Draw background rectangle behind text
        rect = canvas.create_rectangle(bbox[0] - pad_x, bbox[1] - pad_y,bbox[2] + pad_x, bbox[3] + pad_y,fill=bg, outline=bg, tags=tags)
        # Bring text above rectangle
        canvas.tag_raise(temp, rect)


    # ---------------- LAYOUT ----------------
    def draw_layout():
        canvas.delete("layout")

        for a in algos:
            bx = base_x[a]

            # Algorithm title
            draw_tag(bx, 50, a, font=("Segoe UI", 20, "bold"))

            for s in states:
                y = ymap[s]

                # State box
                canvas.create_rectangle(
                    bx - BOX_HALF_WIDTH, y - BOX_HALF_HEIGHT,
                    bx + BOX_HALF_WIDTH, y + BOX_HALF_HEIGHT,
                    width=3,
                    outline="white",
                    tags="layout"
                )

                # State label
                draw_tag(bx, y - 45, s, font=("Segoe UI", 11, "bold"))


        # Keep layout above background
        canvas.tag_raise("layout")
        canvas.tag_raise("token")

    draw_layout()

    # ---------------- POSITION SYSTEM ----------------
    def pos(bx, state, slot):
        col = slot % TOKENS_PER_ROW
        x = bx - (TOKENS_PER_ROW - 1) * TOKEN_SPACING_X // 2 + col * TOKEN_SPACING_X
        y = ymap[state]
        return x, y

    # ---------------- TOKEN CREATION ----------------
    def create_tokens():
        for algo in algos:
            bx = base_x[algo]
            for i, p in enumerate(plist):
                col = i % TOKENS_PER_ROW

                x = bx - (TOKENS_PER_ROW - 1) * TOKEN_SPACING_X // 2 + col * TOKEN_SPACING_X
                y = ymap["NEW"]

                t = canvas.create_oval(
                    x - 20, y - 20,
                    x + 20, y + 20,
                    fill="#e056fd",
                    outline="black",
                    width=2,
                    tags="token"
                )
                lbl = canvas.create_text(
                    x, y,
                    text=p.pid,
                    font=("Segoe UI", 10, "bold"),
                    fill="black",
                    tags="token"
                )

                tokens[algo][p.pid] = (t, lbl)

        canvas.tag_raise("token")

    create_tokens()

    # ---------------- SMOOTH MOTION ----------------
    def move_step(token, tx, ty, steps=STEPS):
        t, lbl = token
        x1, y1, x2, y2 = canvas.coords(t)
        sx, sy = (x1 + x2) / 2, (y1 + y2) / 2
        dx = tx - sx
        dy = ty - sy

        def step(i=0):
            if i > steps:
                return

            k = i / steps
            cx = sx + dx * k
            cy = sy + dy * k

            x1, y1, x2, y2 = canvas.coords(t)
            px = (x1 + x2) / 2
            py = (y1 + y2) / 2

            canvas.move(t, cx - px, cy - py)
            canvas.move(lbl, cx - px, cy - py)

            canvas.tag_raise("token")

            canvas.after(SPEED, lambda: step(i + 1))

        step()

    # ---------------- FCFS ----------------
    def play_fcfs():
        state_slots = {"READY": 0, "TERMINATED": 0}
        seq = sorted(plist, key=lambda x: x.at)

        def run(i=0):
            if i >= len(seq):
                return

            p = seq[i]

            slot_r = state_slots["READY"]
            state_slots["READY"] += 1

            move_step(tokens["FCFS"][p.pid],
                      *pos(base_x["FCFS"], "READY", slot_r))

            canvas.after(700, lambda:
                move_step(tokens["FCFS"][p.pid],
                          *pos(base_x["FCFS"], "RUNNING", 0)))

            def finish():
                slot_t = state_slots["TERMINATED"]
                state_slots["TERMINATED"] += 1

                move_step(tokens["FCFS"][p.pid],
                          *pos(base_x["FCFS"], "TERMINATED", slot_t))

                canvas.after(500, lambda: run(i + 1))

            canvas.after(700 + int(p.bt * 300), finish)

        run()

    # ---------------- ROUND ROBIN ----------------
    def play_rr():
        from collections import deque

        state_slots = {"READY": 0, "BLOCKED": 0, "TERMINATED": 0}

        Q = 3
        queue = deque(plist)
        rem = {p.pid: p.bt for p in plist}

        def run():
            if not queue:
                return

            p = queue.popleft()
            pid = p.pid

            move_step(tokens["RR"][pid],
                      *pos(base_x["RR"], "RUNNING", 0))

            def finish():
                rem[pid] -= Q

                if rem[pid] > 0:
                    slot_b = state_slots["BLOCKED"]
                    state_slots["BLOCKED"] += 1

                    move_step(tokens["RR"][pid],
                              *pos(base_x["RR"], "BLOCKED", slot_b))

                    def back_ready():
                        slot_r = state_slots["READY"]
                        state_slots["READY"] += 1

                        move_step(tokens["RR"][pid],
                                  *pos(base_x["RR"], "READY", slot_r))
                        queue.append(p)

                    canvas.after(400, back_ready)
                else:
                    slot_t = state_slots["TERMINATED"]
                    state_slots["TERMINATED"] += 1

                    move_step(tokens["RR"][pid],
                              *pos(base_x["RR"], "TERMINATED", slot_t))
                    del rem[pid]

                canvas.after(800, run)

            canvas.after(800, finish)

        run()

    # ---------------- ML ----------------
    def play_ml():
        state_slots = {"READY": 0, "TERMINATED": 0}

        ready = []
        completed = set()
        plist_sorted = sorted(plist, key=lambda x: x.at)
        time_now = 0

        def run():
            nonlocal time_now

            for p in plist_sorted:
                if p.pid not in completed and p not in ready and p.at <= time_now:
                    ready.append(p)

                    slot_r = state_slots["READY"]
                    state_slots["READY"] += 1

                    move_step(tokens["ML"][p.pid],
                              *pos(base_x["ML"], "READY", slot_r))

            if not ready:
                time_now += 1
                canvas.after(300, run)
                return

            short = [p for p in ready if should_split_ml(p) == 0]
            long = [p for p in ready if should_split_ml(p) == 1]

            p = min(short if short else long, key=lambda x: x.remaining)
            ready.remove(p)

            run_time = min(get_ml_quantum(p), p.remaining)

            move_step(tokens["ML"][p.pid],
                      *pos(base_x["ML"], "RUNNING", 0))

            def finish():
                nonlocal time_now
                p.remaining -= run_time
                time_now += run_time

                if p.remaining > 0:
                    ready.append(p)

                    slot_r = state_slots["READY"]
                    state_slots["READY"] += 1

                    move_step(tokens["ML"][p.pid],
                              *pos(base_x["ML"], "READY", slot_r))
                else:
                    completed.add(p.pid)

                    slot_t = state_slots["TERMINATED"]
                    state_slots["TERMINATED"] += 1

                    move_step(tokens["ML"][p.pid],
                              *pos(base_x["ML"], "TERMINATED", slot_t))

                if len(completed) < len(plist):
                    canvas.after(700, run)

            canvas.after(700 + int(run_time * 250), finish)

        run()

    # ---------------- BUTTONS ----------------
    tk.Button(win, text="▶ FCFS", font=("Segoe UI", 12, "bold"),
              command=play_fcfs).place(x=300, y=650)

    tk.Button(win, text="▶ RR", font=("Segoe UI", 12, "bold"),
              command=play_rr).place(x=750, y=650)

    tk.Button(win, text="▶ ML", font=("Segoe UI", 12, "bold"),
              command=play_ml).place(x=1200, y=650)

# =========================
# 🖥️ MAIN GUI
# =========================
root = tk.Tk()
root.title("🔥 AI CPU Scheduling Simulator (ML + Live States)")
root.geometry("1000x650")

# =========================
# 🎨 GLOBAL UI STYLE
# =========================
DEFAULT_FONT = ("Segoe UI", 11, "bold")
HEADER_FONT = ("Segoe UI", 12, "bold")
BIG_BUTTON_FONT = ("Segoe UI", 11, "bold")

style = ttk.Style()
style.configure("Treeview", font=("Segoe UI", 11), rowheight=28)
style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"))

# =========================
# 🖼️ RESPONSIVE BACKGROUND
# =========================
bg_original = Image.open(r"C:\SIMATS\capstone\os\cpu-scheduling.webp")
bg_photo = None
bg_label = tk.Label(root)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.lower()

def resize_bg(event):
    global bg_photo
    resized = bg_original.resize((event.width, event.height))
    bg_photo = ImageTk.PhotoImage(resized)
    bg_label.config(image=bg_photo)

root.bind("<Configure>", resize_bg)

# =========================
# GRID
# =========================
for i in range(3):
    root.grid_columnconfigure(i, weight=1)

# =========================
# HEADERS
# =========================
tk.Label(root, text="Arrival", bg="black", fg="white", font=HEADER_FONT).grid(row=0, column=1, pady=5)
tk.Label(root, text="Burst", bg="black", fg="white", font=HEADER_FONT).grid(row=0, column=2, pady=5)

# =========================
# INPUT
# =========================
entries = []
for i in range(4):
    tk.Label(root, text=f"P{i+1}", bg="black", fg="white", font=DEFAULT_FONT)\
        .grid(row=i+1, column=0, pady=4)

    at = tk.Entry(root, width=10, justify="center", font=DEFAULT_FONT)
    bt = tk.Entry(root, width=10, justify="center", font=DEFAULT_FONT)
    at.grid(row=i+1, column=1, pady=6)
    bt.grid(row=i+1, column=2, pady=6)
    entries.append((at, bt))

# =========================
# 🆚 COMPARE
# =========================
def compare_algorithms():
    p = get_processes()
    if not p:
        return
    table.delete(*table.get_children())
    for name, algo in [("FCFS", fcfs), ("RR", round_robin), ("V-RTS-ML", v_rts_ml_scheduler)]:
        avg_wt, avg_tat = compute_metrics(p, algo)
        table.insert("", "end", values=(name, f"{avg_wt:.2f}", f"{avg_tat:.2f}"))

tk.Button(root, text="🆚 Compare Algorithms (Table)",
    bg="#607d8b", fg="white", font=BIG_BUTTON_FONT,
    command=compare_algorithms
).grid(row=7, column=1, ipadx=40, ipady=10, pady=8)

# =========================
# 📦 MODULE 1
# =========================
module1_frame = tk.Frame(root, bg="black")
module1_visible = False

def toggle_module1():
    global module1_visible
    if module1_visible:
        module1_frame.grid_remove()
        module1_visible = False
    else:
        module1_frame.grid(row=9, column=0, columnspan=3, pady=10)
        module1_visible = True

tk.Button(root, text="📦 MODULE 3 (FCFS + RR + Performance)",
    bg="#00bcd4", font=BIG_BUTTON_FONT,
    command=toggle_module1
).grid(row=8, column=1, ipadx=40, ipady=10, pady=8)

tk.Button(module1_frame, text="FCFS Gantt", width=18, font=BIG_BUTTON_FONT,
    command=lambda: draw_gantt(fcfs(copy.deepcopy(get_processes())), "FCFS Gantt")
).grid(row=0, column=0, padx=20, pady=5)

tk.Button(module1_frame, text="RR Gantt", width=18, font=BIG_BUTTON_FONT,
    command=lambda: draw_gantt(round_robin(copy.deepcopy(get_processes())), "RR Gantt")
).grid(row=0, column=1, padx=20, pady=5)

tk.Button(module1_frame, text="📊 Performance Graph", width=22, font=BIG_BUTTON_FONT,
    command=run_performance_analysis
).grid(row=0, column=2, padx=20, pady=5)

# =========================
# 📦 MODULE 2
# =========================
module2_frame = tk.Frame(root, bg="black")
module2_visible = False

def toggle_module2():
    global module2_visible
    if module2_visible:
        module2_frame.grid_remove()
        module2_visible = False
    else:
        module2_frame.grid(row=11, column=0, columnspan=3, pady=10)
        module2_visible = True

tk.Button(root, text="📦 MODULE 2 (RTS Only)",
    bg="#9c27b0", fg="white", font=BIG_BUTTON_FONT,
    command=toggle_module2
).grid(row=10, column=1, ipadx=40, ipady=10, pady=8)

tk.Button(module2_frame, text="V-RTS-ML Gantt", width=25, font=BIG_BUTTON_FONT,
    command=lambda: draw_gantt(v_rts_ml_scheduler(copy.deepcopy(get_processes())), "V-RTS-ML Gantt")
).grid(row=0, column=0, padx=20, pady=5)

# =========================
# 🎮 MODULE 4
# =========================
module4_frame = tk.Frame(root, bg="black")
module4_visible = False

def toggle_module4():
    global module4_visible
    if module4_visible:
        module4_frame.grid_remove()
        module4_visible = False
    else:
        module4_frame.grid(row=13, column=0, columnspan=3, pady=10)
        module4_visible = True

tk.Button(root, text="🎮 MODULE 4 (Simulation Lab)",
    bg="#ff9800", font=BIG_BUTTON_FONT,
    command=toggle_module4
).grid(row=12, column=1, ipadx=40, ipady=10, pady=8)

tk.Button(module4_frame, text="🎮 Start Live Simulation", width=30, bg="orange",
    font=BIG_BUTTON_FONT,
    command=side_by_side_simulation
).grid(row=0, column=0, padx=20, pady=5)

# =========================
# 📊 TABLE
# =========================
table = ttk.Treeview(root, columns=("Method", "Avg WT", "Avg TAT"),
    show="headings", height=6
)
table.heading("Method", text="Method")
table.heading("Avg WT", text="Avg WT")
table.heading("Avg TAT", text="Avg TAT")
table.grid(row=16, column=0, columnspan=3, padx=20, pady=15)

# =========================
# 🚀 RUN
# =========================
root.mainloop()
