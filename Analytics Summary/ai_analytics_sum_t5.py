from sentence_transformers import SentenceTransformer, util
from dateutil import parser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import sqlite3
import torch
import re


# --- Configuration ---
DB_PATH = "mini_ai_sumgen.db"
model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t2t_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
t2t_pipeline = pipeline("text2text-generation", model=t2t_model, tokenizer=tokenizer)


# --- Static Filter Keywords ---
TYPES = ["Extension", "Research"]
STATUSES = ["Pending", "Ongoing", "Completed", "New", "Accepted", "Rejected"]
COLLEGES = [
    "College of Arts and Humanities", "College of Business and Accountancy",
    "College of Criminal Justice Education", "College of Engineering Architecture and Technology",
    "College of Hospitality Management and Tourism", "College of Nursing and Health Sciences",
    "College of Sciences", "College of Teacher Education"
]


# --- Intent Templates ---
TEMPLATES = [
    {"template": "Number of projects", "function": "get_number_of_projects"},
    {"template": "College budget allocation", "function": "get_budget_distribution"},
    {"template": "Distribution of agenda types", "function": "get_agenda_distribution"},
    {"template": "Number of trained individuals", "function": "get_trained_individuals"},
    {"template": "Client request counts", "function": "get_client_requests"},
    {"template": "SDG project breakdown", "function": "get_sdg_distribution"},
]
TEMPLATE_EMBEDDINGS = model.encode([t["template"] for t in TEMPLATES], convert_to_tensor=True)


# --- Fuzzy Function Selector ---
def detect_function(user_input):
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    results = util.semantic_search(input_embedding, TEMPLATE_EMBEDDINGS, top_k=1)
    idx = results[0][0]["corpus_id"]
    return TEMPLATES[idx]["function"]


# --- Extract Timeline ---
def parse_timeline(timeline_type, value):
    try:
        if timeline_type == "year":
            year = int(value)
            return year, None
        elif timeline_type == "month":
            dt = parser.parse(value, fuzzy=True)
            return dt.year, dt.month
    except:
        pass
    return None, None


# --- Extract Filters from Input ---
def extract_filters(user_input):
    text = user_input.lower()
    type_filter = next((t for t in TYPES if t.lower() in text), None)
    status_filter = next((s for s in STATUSES if s.lower() in text), None)
    college_filter = next((c for c in COLLEGES if c.lower() in text), None)
    return {
        "type": type_filter,
        "status": status_filter,
        "college": college_filter
    }


# --- Summary Formatter ---
def summarize_output(data_dict, label):
    if not data_dict:
        return "No data found."
    if "error" in data_dict:
        return f"⚠️ Error: {data_dict['error']}"
    summary = f"\n📊 Summary for: \"{label}\"\n"
    summary += "\n".join([f"• {k}: {v}" for k, v in data_dict.items()])
    return summary


# --- Dispatcher ---
def dispatch(func_name, year, month=None, input_text=None):
    filters = extract_filters(input_text or "")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        if func_name == "get_number_of_projects":
            return get_number_of_projects(cursor, year, month, filters["type"], filters["status"], filters["college"])

        elif func_name == "get_budget_distribution":
            return get_budget_distribution(cursor, year, filters["college"])

        elif func_name == "get_agenda_distribution":
            return get_agenda_distribution(cursor, year, month, filters["college"])

        elif func_name == "get_trained_individuals":
            return get_trained_individuals(cursor, year, month, filters["college"])

        elif func_name == "get_client_requests":
            return get_client_requests(cursor, year, month, filters["status"])

        elif func_name == "get_sdg_distribution":
            return get_sdg_distribution(cursor, year, month, filters["college"])

        return {"error": "Unknown function"}


# --- Individual Analytics Functions ---
def get_number_of_projects(cursor, year, month=None, type_filter=None, status_filter=None, college=None):
    query = """
        SELECT type, status, colleges.name, COUNT(*)
        FROM projects
        JOIN project_team ON project_team.project_id = projects.id AND team_role = 'Leader'
        JOIN users ON users.id = project_team.user_id
        JOIN colleges ON colleges.id = users.college_id
        WHERE strftime('%Y-%m', projects.start_date) LIKE ?
    """
    params = [f"{year}-{month:02d}%" if month else f"{year}%"]

    if type_filter:
        query += " AND type = ?"
        params.append(type_filter)
    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)
    if college:
        query += " AND colleges.name = ?"
        params.append(college)

    query += " GROUP BY type, status, colleges.name"
    cursor.execute(query, params)
    results = cursor.fetchall()

    # Map to correct label based on filter combination
    formatted = {}

    for t, s, c, count in results:
        # All three filters — optional total only
        if type_filter and status_filter and college:
            key = "Total"
            formatted[key] = formatted.get(key, 0) + count

        # Two filters: show the one not mentioned
        elif type_filter and status_filter:
            key = f"{c}"
            formatted[key] = formatted.get(key, 0) + count
        elif type_filter and college:
            key = f"{s}"
            formatted[key] = formatted.get(key, 0) + count
        elif status_filter and college:
            key = f"{t}"
            formatted[key] = formatted.get(key, 0) + count

        # One filter: show the other two
        elif type_filter:
            key = f"{c} | {s}"
            formatted[key] = formatted.get(key, 0) + count
        elif status_filter:
            key = f"{c} | {t}"
            formatted[key] = formatted.get(key, 0) + count
        elif college:
            key = f"{t} | {s}"
            formatted[key] = formatted.get(key, 0) + count

        # No filters: show all
        else:
            key = f"{c} | {t} | {s}"
            formatted[key] = formatted.get(key, 0) + count

    return formatted


def get_budget_distribution(cursor, year, college=None):
    query = """
        SELECT colleges.name, SUM(college_budgets.allocated_budget)
        FROM college_budgets
        JOIN colleges ON colleges.id = college_budgets.college_id
        WHERE year = ?
    """
    params = [year]
    if college:
        query += " AND colleges.name = ?"
        params.append(college)
    query += " GROUP BY colleges.name"
    cursor.execute(query, params)
    return dict(cursor.fetchall())


def get_agenda_distribution(cursor, year, month=None, college=None):
    query = """
        SELECT agendas.name, COUNT(*)
        FROM projects
        JOIN agendas ON agendas.id = projects.agenda_id
        JOIN project_team ON project_team.project_id = projects.id AND team_role = 'Leader'
        JOIN users ON users.id = project_team.user_id
        JOIN colleges ON colleges.id = users.college_id
        WHERE strftime('%Y-%m', projects.start_date) LIKE ?
    """
    params = [f"{year}-{month:02d}%" if month else f"{year}%"]
    if college:
        query += " AND colleges.name = ?"
        params.append(college)
    query += " GROUP BY agendas.name"
    cursor.execute(query, params)
    return dict(cursor.fetchall())


def get_trained_individuals(cursor, year, month=None, college=None):
    query = """
        SELECT colleges.name, SUM(submission_responses.trained_individuals)
        FROM submission_responses
        JOIN submission_requirements ON sub_req_id = submission_requirements.id
        JOIN projects ON submission_requirements.project_id = projects.id
        JOIN project_team ON project_team.project_id = projects.id AND team_role = 'Leader'
        JOIN users ON users.id = project_team.user_id
        JOIN colleges ON colleges.id = users.college_id
        WHERE strftime('%Y-%m', submission_responses.submitted_at) LIKE ?
    """
    params = [f"{year}-{month:02d}%" if month else f"{year}%"]

    if college:
        query += " AND colleges.name = ?"
        params.append(college)

    query += " GROUP BY colleges.name"
    cursor.execute(query, params)
    return dict(cursor.fetchall())


def get_client_requests(cursor, year, month=None, status=None):
    query = """
        SELECT status, COUNT(*) FROM client_requests
        WHERE strftime('%Y-%m', created_at) LIKE ?
    """
    params = [f"{year}-{month:02d}%" if month else f"{year}%"]

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " GROUP BY status"
    cursor.execute(query, params)
    result = dict(cursor.fetchall())

    # If specific status was asked, filter others out
    if status:
        result = {status: result.get(status, 0)}
    return result


def get_sdg_distribution(cursor, year, month=None, college=None):
    query = """
        SELECT sdgs.title, COUNT(*)
        FROM sdg_project
        JOIN sdgs ON sdgs.id = sdg_project.sdgs_id
        JOIN projects ON projects.id = sdg_project.project_id
        JOIN project_team ON project_team.project_id = projects.id AND team_role = 'Leader'
        JOIN users ON users.id = project_team.user_id
        JOIN colleges ON colleges.id = users.college_id
        WHERE strftime('%Y-%m', projects.start_date) LIKE ?
    """
    params = [f"{year}-{month:02d}%" if month else f"{year}%"]
    if college:
        query += " AND colleges.name = ?"
        params.append(college)
    query += " GROUP BY sdgs.title"
    cursor.execute(query, params)
    return dict(cursor.fetchall())


# --- Text-to-Text NLG ---
def generate_nlg_summary(data_dict, user_query):
    if not data_dict:
        return "No data to summarize."
    
    bullet_points = "\n".join([f"- {k}: {v}" for k, v in data_dict.items()])
    prompt = f"""Generate a brief, natural-sounding report about:
{user_query}
{bullet_points}"""
    
    result = t2t_pipeline(prompt, max_length=120, do_sample=True, top_k=50, temperature=0.7)[0]["generated_text"]
    return result.strip()

# --- AI Summary Entry Point ---
def ai_summary(field_of_focus, timeline_type, value):
    func = detect_function(field_of_focus)
    year, month = parse_timeline(timeline_type, value)
    if not year:
        return "⚠️ Invalid timeline input."
    
    results = dispatch(func, year, month, field_of_focus)
    readable = summarize_output(results, f"{field_of_focus} in {value}")
    generated = generate_nlg_summary(results, f"{field_of_focus} in {value}")
    
    return f"{readable}\n\n📝 Generated Summary:\n{generated}"

# === Example Usage ===
if __name__ == "__main__":
    print(ai_summary("Extension Projects", "year", "2024"))                                                         # Project with Type
    print(ai_summary("Ongoing Extension Projects", "year", "2024"))                                                 # Project with Type and Status
    print(ai_summary("Projects in College of Engineering Architecture and Technology", "year", "2024"))             # Project with Type, Status, and College
    print(ai_summary("Budget Distribution", "year", "2022"))                                                        # Budget Distribution
    print(ai_summary("Budget of College of Sciences", "year", "2022"))                                              # Budget Distribution with College
    print(ai_summary("Distribution of Agenda Types", "year", "2024"))                                               # Agenda Distribution
    print(ai_summary("Trained Individuals", "year", "2022"))                                                        # Trained Individuals
    print(ai_summary("Trained Individuals in College of Nursing and Health Sciences", "month", "March 2022"))       # Trained Individuals with College
    print(ai_summary("Client Request", "year", "2020"))                                                             # Client Requests
    print(ai_summary("Rejected Client Requests", "month", "April 2020"))                                            # Client Requests with Status                           
    print(ai_summary("SDG Project Breakdown", "year", "2023"))                                                      # SDG Distribution   
