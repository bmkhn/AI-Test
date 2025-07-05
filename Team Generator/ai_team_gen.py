import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

def generate_team(keywords, campus_filter, college_filter, num_participants):
    def normalize(series):
        if series.max() != series.min():
            return (series - series.min()) / (series.max() - series.min())
        return np.zeros_like(series)

    # --- Configuration ---
    DB_PATH = "mini_ai_teamgen.db"
    DEGREE_WEIGHT = 0.5
    EXPERTISE_WEIGHT = 0.3
    RATING_WEIGHT = 0.15
    AVAILABILITY_WEIGHT = 0.05

    # --- Connect to DB and Fetch Data ---
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

    # --- Apply Filters ---
    if campus_filter:
        users_df = users_df[users_df["campus"] == campus_filter]
    if college_filter:
        users_df = users_df[users_df["college_name"] == college_filter]

    if users_df.empty:
        print("❌ No users found after applying filters.")
        return

    # --- Compute Semantic Similarity ---
    model = SentenceTransformer('all-MiniLM-L6-v2')
    keyword_embedding = model.encode(keywords, convert_to_tensor=True)
    degree_embeddings = model.encode(users_df["degree"].tolist(), convert_to_tensor=True)
    expertise_embeddings = model.encode(users_df["expertise"].tolist(), convert_to_tensor=True)

    degree_scores = util.cos_sim(keyword_embedding, degree_embeddings)[0].cpu().numpy()
    expertise_scores = util.cos_sim(keyword_embedding, expertise_embeddings)[0].cpu().numpy()

    # --- Normalize Ratings and Availability ---
    normalized_ratings = normalize(users_df["avg_rating"].fillna(0))
    normalized_ongoing = normalize(users_df["ongoing_projects"].fillna(0))
    availability_scores = 1 - normalized_ongoing

    # --- Compute Final Score ---
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

    # --- Sort and Show Results ---
    top_users = users_df.sort_values(by="final_score", ascending=False).head(num_participants).reset_index(drop=True)

    print("\nTop AI-Matched Participants:")
    print(top_users[[
        "name",
        "degree", "expertise",      # Degree and Expertise
        "degree_score",             # Degree Relevance Score
        "expertise_score",          # Expertise Relevance Score
        "normalized_rating",        # Normalized Rating [Evaluation Scores]
        "availability_score",       # Availability Score
        "final_score"
    ]])

# -- Sample Usage --
if __name__ == "__main__":
    generate_team(
        keywords="artificial intelligence",
        campus_filter=None,
        college_filter=None,
        num_participants=5
    )