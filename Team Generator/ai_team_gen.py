import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
DB_PATH = "mini_ai_teamgen.db"
KEYWORDS = "artificial intelligence"         # Keywords to match degree AND expertise against
CAMPUS_FILTER = None            # e.g., "Tinuigiban" or None
COLLEGE_FILTER = None           # e.g., "College of Sciences" or None
NUM_PARTICIPANTS = 5

DEGREE_WEIGHT = 0.5             # Placeholder weight for degree relevance
EXPERTISE_WEIGHT = 0.3          # Placeholder weight for expertise relevance
RATING_WEIGHT = 0.15            # Placeholder weight for evaluation performance
AVAILABILITY_WEIGHT = 0.05      # Placeholder weight for availability


def normalize(series):
    if series.max() != series.min():
        return (series - series.min()) / (series.max() - series.min())
    return np.zeros_like(series)


# --- 1. Connect and Fetch Data ---
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


# --- 2. Filter by Campus or College ---
if CAMPUS_FILTER:
    users_df = users_df[users_df["campus"] == CAMPUS_FILTER]
if COLLEGE_FILTER:
    users_df = users_df[users_df["college_name"] == COLLEGE_FILTER]

if users_df.empty:
    print("❌ No users found after applying filters.")
    exit()


# --- 3. Compute Semantic Similarity for Degree and Expertise ---
model = SentenceTransformer('all-MiniLM-L6-v2')
keyword_embedding = model.encode(KEYWORDS, convert_to_tensor=True)

degree_embeddings = model.encode(users_df["degree"].tolist(), convert_to_tensor=True)
expertise_embeddings = model.encode(users_df["expertise"].tolist(), convert_to_tensor=True)

degree_scores = util.cos_sim(keyword_embedding, degree_embeddings)[0].cpu().numpy()
expertise_scores = util.cos_sim(keyword_embedding, expertise_embeddings)[0].cpu().numpy()


# --- 4. Normalize Scores ---
normalized_ratings = normalize(users_df["avg_rating"].fillna(0))
normalized_ongoing = normalize(users_df["ongoing_projects"].fillna(0))
availability_scores = 1 - normalized_ongoing  # Fewer ongoing = more available


# --- 5. Compute Final Score Based on Hierarchy ---
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


# --- 6. Select Top N ---
top_users = users_df.sort_values(by="final_score", ascending=False).head(NUM_PARTICIPANTS).reset_index(drop=True)


# --- 7. Display Result ---
print("\nTop AI-Matched Participants:")
print(top_users[[
    # "id", 
    "name", 
    # "campus", "college_name", 
    "degree", "expertise",                              # Degree and Expertise
    "degree_score",                                     # Degree Relevance Score
    "expertise_score",                                  # Expertise Relevance Score
    # "total_projects", "ongoing_projects",
    "normalized_rating",                                # Normalized Rating
    "availability_score",                               # Availability Score
    "final_score" 
]])