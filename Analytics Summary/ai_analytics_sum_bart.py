from sentence_transformers import SentenceTransformer, util
import dateparser
import sqlite3
from transformers import pipeline
import random
import warnings

# --- Sentence Transformer Model ---
model = SentenceTransformer('all-MiniLM-L6-v2') # another option all-mpnet-base-v2

# --- BART Summarization Pipeline ---
warnings.filterwarnings("ignore", category=UserWarning)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- DB Config ---
DB_PATH = "mini_ai_sumgen.db"

# --- Constants ---
TYPES = ["Extension", "Research"]
STATUSES = ["Pending", "Ongoing", "Completed", "New", "Accepted", "Rejected"]
COLLEGES = [
    "College of Arts and Humanities", "College of Business and Accountancy",
    "College of Criminal Justice Education", "College of Engineering Architecture and Technology",
    "College of Hospitality Management and Tourism", "College of Nursing and Health Sciences",
    "College of Sciences", "College of Teacher Education"
]

TEMPLATES = [
    {"template": "Number of projects", "function": "get_number_of_projects"},
    {"template": "College budget allocation", "function": "get_budget_distribution"},
    {"template": "Distribution of agenda types", "function": "get_agenda_distribution"},
    {"template": "Number of trained individuals", "function": "get_trained_individuals"},
    {"template": "Client request counts", "function": "get_client_requests"},
    {"template": "SDG project breakdown", "function": "get_sdg_distribution"},
]
TEMPLATE_EMBEDDINGS = model.encode([t["template"] for t in TEMPLATES], convert_to_tensor=True)

# --- Intent Detection ---
def detect_function(user_input: str):
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    results = util.semantic_search(input_embedding, TEMPLATE_EMBEDDINGS, top_k=1)
    idx = results[0][0]["corpus_id"]
    return TEMPLATES[idx]["function"]

# --- Timeline Parsing ---
def parse_timeline(timeline_type: str, value: str):
    dt = dateparser.parse(value, settings={"PREFER_DATES_FROM": "past"})
    if not dt:
        return None, None
    return (dt.year, None) if timeline_type == "year" else (dt.year, dt.month)

# --- Semantic Filter Extraction ---
def extract_semantic_filter(user_input: str, label_list: list):
    embeddings = model.encode(label_list, convert_to_tensor=True)
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    results = util.semantic_search(input_embedding, embeddings, top_k=1)
    return label_list[results[0][0]["corpus_id"]]

def extract_filters(user_input: str):
    filters = {"type": None, "status": None, "college": None}
    lowered = user_input.lower()
    if any(t.lower() in lowered for t in TYPES):
        filters["type"] = extract_semantic_filter(user_input, TYPES)
    if any(s.lower() in lowered for s in STATUSES):
        filters["status"] = extract_semantic_filter(user_input, STATUSES)
    if any(c.lower() in lowered for c in COLLEGES):
        filters["college"] = extract_semantic_filter(user_input, COLLEGES)
    return filters

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

# --- Analytics Functions ---
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

    formatted = {}
    for t, s, c, count in results:
        if type_filter and status_filter and college:
            key = "Total"
        elif type_filter and status_filter:
            key = c
        elif type_filter and college:
            key = s
        elif status_filter and college:
            key = t
        elif type_filter:
            key = f"{c} | {s}"
        elif status_filter:
            key = f"{c} | {t}"
        elif college:
            key = f"{t} | {s}"
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
    return dict(cursor.fetchall())

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


# --- Data Formatting for BART ---
# def format_data_for_bart(data_dict, user_query):
#     if not data_dict:
#         return "There is no data available to summarize."

#     sentences = []
#     for key, value in data_dict.items():
#         parts = [p.strip() for p in key.split("|")]

#         # Projects summary
#         if "project" in user_query.lower():
#             if len(parts) == 3:
#                 college, type_, status = parts
#                 templates = [
#                     f"In {user_query}, {college} had {value} {status.lower()} {type_.lower()} project{'s' if value != 1 else ''}.",
#                     f"{college} implemented {value} {status.lower()} {type_.lower()} project{'s' if value != 1 else ''} in {user_query}.",
#                     f"{value} {type_.lower()} project{'s' if value != 1 else ''} were marked as {status.lower()} under {college}."
#                 ]
#             elif len(parts) == 2:
#                 part1, part2 = parts
#                 templates = [
#                     f"{part1} had {value} {part2.lower()} project{'s' if value != 1 else ''}.",
#                     f"{value} projects in {part1} were labeled as {part2.lower()}.",
#                     f"{part1} reported {value} {part2.lower()} projects."
#                 ]
#             else:
#                 templates = [f"{key} had {value} project{'s' if value != 1 else ''}."]
        
#         # Budget
#         elif "budget" in user_query.lower():
#             templates = [
#                 f"{key} received a budget allocation of ₱{value:,}.",
#                 f"The total budget for {key} was ₱{value:,}.",
#                 f"₱{value:,} was designated for {key} in {user_query}."
#             ]

#         # Trained individuals
#         elif "trained" in user_query.lower() or "individuals" in user_query.lower():
#             templates = [
#                 f"{value} individual{'s' if value != 1 else ''} were trained under {key}.",
#                 f"{key} trained a total of {value} person{'s' if value != 1 else ''}.",
#                 f"Training efforts by {key} reached {value} participant{'s' if value != 1 else ''}."
#             ]

#         # Agenda
#         elif "agenda" in user_query.lower():
#             templates = [
#                 f"{key} appeared in the agenda {value} time{'s' if value != 1 else ''}.",
#                 f"{value} items were recorded under {key} in the agenda.",
#                 f"The agenda included {key} on {value} occasion{'s' if value != 1 else ''}."
#             ]

#         # SDG
#         elif "sdg" in user_query.lower():
#             templates = [
#                 f"{value} projects aligned with the SDG goal: {key}.",
#                 f"The Sustainable Development Goal '{key}' had {value} related project{'s' if value != 1 else ''}.",
#                 f"{value} initiatives addressed the goal: {key}."
#             ]

#         # Client requests
#         elif "client request" in user_query.lower():
#             templates = [
#                 f"There were {value} client request{'s' if value != 1 else ''} marked as {key.lower()}.",
#                 f"{value} request{'s' if value != 1 else ''} were {key.lower()} during the period.",
#                 f"The system recorded {value} {key.lower()} client request{'s' if value != 1 else ''}."
#             ]

#         else:
#             templates = [f"{key} had {value} instance{'s' if value != 1 else ''}."]

#         sentences.append(random.choice(templates))

#     return " ".join(sentences)


# --- NLG Summary Generation ---

def generate_nlg_summary(data_dict, user_query):
    if not data_dict:
        return "No data available to summarize."
    if "error" in data_dict:
        return f"⚠️ Error: {data_dict['error']}"

    # Try to guess analytic type
    query_lower = user_query.lower()
    if "budget" in query_lower:
        data_type = "budget"
    elif "trained" in query_lower or "individuals" in query_lower:
        data_type = "training"
    elif "agenda" in query_lower:
        data_type = "agenda"
    elif "client request" in query_lower:
        data_type = "client_requests"
    elif "sdg" in query_lower:
        data_type = "sdg"
    else:
        data_type = "projects"

    # Create natural language sentences per data_type
    structured_lines = []
    for k, v in data_dict.items():
        if data_type == "budget":
            structured_lines.append(f"{k} received a budget allocation of ₱{int(v):,}.")
        elif data_type == "training":
            structured_lines.append(f"{k} trained {int(v)} individual{'s' if v != 1 else ''}.")
        elif data_type == "agenda":
            structured_lines.append(f"{k} appeared {int(v)} time{'s' if v != 1 else ''} in the agenda.")
        elif data_type == "client_requests":
            structured_lines.append(f"{int(v)} client request{'s' if v != 1 else ''} were marked as {k.lower()}.")
        elif data_type == "sdg":
            structured_lines.append(f"{k} had {int(v)} associated project{'s' if v != 1 else ''}.")
        else:  # For projects and mixed filters
            parts = [p.strip() for p in k.split("|")]
            if len(parts) == 3:
                structured_lines.append(f"{parts[0]} had {v} {parts[2].lower()} {parts[1].lower()} project{'s' if v != 1 else ''}.")
            elif len(parts) == 2:
                structured_lines.append(f"{parts[0]} had {v} {parts[1].lower()} project{'s' if v != 1 else ''}.")
            else:
                structured_lines.append(f"{k} had {v} project{'s' if v != 1 else ''}.")

    # Create full prompt
    prompt = (
        f"User asked: \"{user_query}\"\n\n"
        f"Here are the details:\n" + "\n".join(f"- {line}" for line in structured_lines)
        + "\n\nWrite a friendly, informative summary:"
    )

    try:
        result = summarizer(prompt, max_length=60, min_length=15, do_sample=False)
        summary = result[0]['summary_text'].strip()
        return summary
    except Exception as e:
        return f"⚠️ Failed to generate summary: {e}"

# --- Entry Point ---
def ai_summary(field_of_focus: str, timeline_type: str, value: str):
    func = detect_function(field_of_focus)
    year, month = parse_timeline(timeline_type, value)
    if not year:
        return "⚠️ Invalid timeline input."
    results = dispatch(func, year, month, field_of_focus)
    readable = summarize_output(results, f"{field_of_focus} in {value}")
    generated = generate_nlg_summary(results, field_of_focus)
    return f"{readable}\n\n📝 Generated Summary:\n{generated}"


# --- Sample Usage ---
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