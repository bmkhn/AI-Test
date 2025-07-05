import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


# --- CONFIGURATION ---
DB_PATH = "mini_ai_teamgen.db"

DEGREE_WEIGHT = 0.5
EXPERTISE_WEIGHT = 0.3
RATING_WEIGHT = 0.15
AVAILABILITY_WEIGHT = 0.05


# --- FUNCTIONS ---
def normalize(series):
    if series.max() != series.min():
        return (series - series.min()) / (series.max() - series.min())
    return np.zeros_like(series)


def generate_team():
    keywords = keyword_entry.get()
    try:
        num_participants = int(num_participants_entry.get())
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number of participants.")
        return

    campus_filter = campus_entry.get().strip() or None
    college_filter = college_entry.get().strip() or None

    # Connect and Fetch Data
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        users.id,
        users.name,
        users.campus,
        colleges.name AS college_name,
        users.degree,
        users.expertise,
        COUNT(DISTINCT pt_all.project_id) AS total_projects,
        SUM(CASE WHEN p.status = 'Ongoing' THEN 1 ELSE 0 END) AS ongoing_projects,
        AVG(e.rating) AS avg_rating
    FROM users
    JOIN colleges ON users.college_id = colleges.id
    LEFT JOIN project_teams pt_all ON users.id = pt_all.user_id
    LEFT JOIN projects p ON pt_all.project_id = p.id
    LEFT JOIN evaluations e ON p.id = e.project_id
    GROUP BY users.id
    """
    users_df = pd.read_sql_query(query, conn)
    conn.close()

    # Apply Filters
    if campus_filter:
        users_df = users_df[users_df["campus"] == campus_filter]
    if college_filter:
        users_df = users_df[users_df["college_name"] == college_filter]

    if users_df.empty:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "❌ No users found after applying filters.")
        return

    # Semantic Similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    keyword_embedding = model.encode(keywords, convert_to_tensor=True)

    degree_embeddings = model.encode(users_df["degree"].tolist(), convert_to_tensor=True)
    expertise_embeddings = model.encode(users_df["expertise"].tolist(), convert_to_tensor=True)

    degree_scores = util.cos_sim(keyword_embedding, degree_embeddings)[0].cpu().numpy()
    expertise_scores = util.cos_sim(keyword_embedding, expertise_embeddings)[0].cpu().numpy()

    # Normalize
    normalized_ratings = normalize(users_df["avg_rating"].fillna(0))
    normalized_ongoing = normalize(users_df["ongoing_projects"].fillna(0))
    availability_scores = 1 - normalized_ongoing

    # Compute Score
    users_df["degree_score"] = degree_scores
    users_df["expertise_score"] = expertise_scores
    users_df["normalized_rating"] = normalized_ratings
    users_df["availability_score"] = availability_scores

    users_df["final_score"] = (
        users_df["degree_score"] * DEGREE_WEIGHT +
        users_df["expertise_score"] * EXPERTISE_WEIGHT +
        users_df["normalized_rating"] * RATING_WEIGHT +
        users_df["availability_score"] * AVAILABILITY_WEIGHT
    )

    # Select Top N
    top_users = users_df.sort_values(by="final_score", ascending=False).head(num_participants).reset_index(drop=True)

    # Display
    result_text.delete("1.0", tk.END)
    for _, row in top_users.iterrows():
        result_text.insert(tk.END, f"Name: {row['name']}\n")
        result_text.insert(tk.END, f"  Degree: {row['degree']} (Score: {row['degree_score']:.2f})\n")
        result_text.insert(tk.END, f"  Expertise: {row['expertise']} (Score: {row['expertise_score']:.2f})\n")
        result_text.insert(tk.END, f"  Rating Score: {row['normalized_rating']:.2f}\n")
        result_text.insert(tk.END, f"  Availability Score: {row['availability_score']:.2f}\n")
        result_text.insert(tk.END, f"  Final Score: {row['final_score']:.3f}\n\n")


# --- TKINTER GUI ---
root = tk.Tk()
root.title("AI Team Generator")

# Layout
frame = ttk.Frame(root, padding=15)
frame.pack(fill="both", expand=True)

ttk.Label(frame, text="Keyword(s):").grid(row=0, column=0, sticky="w")
keyword_entry = ttk.Entry(frame, width=40)
keyword_entry.insert(0, "AI, Machine Learning, Data Science")
keyword_entry.grid(row=0, column=1)

ttk.Label(frame, text="Number of Participants:").grid(row=1, column=0, sticky="w")
num_participants_entry = ttk.Entry(frame, width=40)
num_participants_entry.insert(0, "5")
num_participants_entry.grid(row=1, column=1)

ttk.Label(frame, text="Campus Filter (optional):").grid(row=2, column=0, sticky="w")
campus_entry = ttk.Entry(frame, width=40)
campus_entry.grid(row=2, column=1)

ttk.Label(frame, text="College Filter (optional):").grid(row=3, column=0, sticky="w")
college_entry = ttk.Entry(frame, width=40)
college_entry.grid(row=3, column=1)

generate_button = ttk.Button(frame, text="Generate Team", command=generate_team)
generate_button.grid(row=4, column=0, columnspan=2, pady=10)

result_text = tk.Text(frame, height=20, width=80, wrap=tk.WORD)
result_text.grid(row=5, column=0, columnspan=2)

root.mainloop()
