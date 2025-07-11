from sentence_transformers import SentenceTransformer, util
from dateutil import parser
import sqlite3
import random

# --- Configuration ---
model = SentenceTransformer('all-MiniLM-L6-v2')
DB_PATH = "mini_ai_sumgen.db"

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


# --- Dispatch Function ---
def dispatch(func_name, year, month=None, filters=None):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        if func_name == "get_number_of_projects":
            return get_number_of_projects(cursor, year, month, filters.get("type"), filters.get("status"), filters.get("college"))

        elif func_name == "get_budget_distribution":
            return get_budget_distribution(cursor, year, filters.get("college"))

        elif func_name == "get_agenda_distribution":
            return get_agenda_distribution(cursor, year, month, filters.get("college"))

        elif func_name == "get_trained_individuals":
            return get_trained_individuals(cursor, year, month, filters.get("college"))

        elif func_name == "get_client_requests":
            return get_client_requests(cursor, year, month, filters.get("status"))

        elif func_name == "get_sdg_distribution":
            return get_sdg_distribution(cursor, year, month, filters.get("college"))

        return {"error": "Unknown function"}


######################################################################################################################################################


# --- Summary Formatter ---
def summarize_output(data_dict, label, filters, func_name):
    if not data_dict:
        return "No data found."
    if "error" in data_dict:
        return f"⚠️ Error: {data_dict['error']}"

    when = label.split(" in ")[-1]
    original_output = f"\n📊 Summary for: \"{label}\"\n"
    for k, v in data_dict.items():
        original_output += f"• {k}: {v}\n"

    summary = ""

    if func_name == "get_number_of_projects":
        total = sum(data_dict.values())
        segments = []

        for k, v in data_dict.items():
            parts = k.split(" | ")
            if len(parts) == 3:
                college, proj_type, status = parts
                segments.append(f"{status} {proj_type.lower()} projects from the {college} ({v})")
            elif len(parts) == 2:
                if filters["type"]:
                    college, status = parts
                    segments.append(f"{status} projects from the {college} ({v})")
                elif filters["status"]:
                    college, proj_type = parts
                    segments.append(f"{proj_type} projects from the {college} ({v})")
                elif filters["college"]:
                    proj_type, status = parts
                    segments.append(f"{status} {proj_type.lower()} projects ({v})")
            else:
                segments.append(f"{k} ({v})")

        intro_variants = [
            f"In {when}, a total of {total} projects were recorded.",
            f"The year {when} saw {total} projects in total.",
            f"A total of {total} projects took place in {when}.",
            f"During {when}, {total} projects were carried out."
        ]
        contrib_variants = [
            f"Top contributors include {', '.join(segments)}.",
            f"Leading contributions came from {', '.join(segments)}.",
            f"Notable entries were recorded by {', '.join(segments)}.",
            f"These projects were primarily from {', '.join(segments)}."
        ]

        summary = f"{random.choice(intro_variants)} {random.choice(contrib_variants)}"

    elif func_name == "get_budget_distribution":
        total = sum(data_dict.values())
        top_alloc = ", ".join([f"{k} ({v:,})" for k, v in data_dict.items()])

        intro = (
            f"In {when}, the total allocated budget reached {total:,} pesos.",
            f"{when} had a total budget allocation of {total:,} pesos.",
            f"A total of {total:,} pesos was distributed in {when}.",
        )
        detail = (
            f"Major allocations went to {top_alloc}.",
            f"Primary recipients include {top_alloc}.",
            f"The largest portions were given to {top_alloc}.",
        )

        if filters["college"]:
            summary = f"In {when}, the total allocated budget in {filters['college']} reached {total:,} pesos, with major allocations to {top_alloc}."
        else:
            summary = f"{random.choice(intro)} {random.choice(detail)}"

    elif func_name == "get_agenda_distribution":
        total = sum(data_dict.values())
        agenda_list = ", ".join([f"{k} ({v})" for k, v in data_dict.items()])
        phrases = [
            f"The agenda distribution for {when} highlights {agenda_list}, covering a total of {total} projects.",
            f"In {when}, {total} projects were spread across agendas like {agenda_list}.",
            f"{total} projects in {when} were categorized under agendas including {agenda_list}.",
        ]
        summary = random.choice(phrases)

    elif func_name == "get_trained_individuals":
        total = sum(data_dict.values())
        trainers = ", ".join([f"{k} ({v})" for k, v in data_dict.items()])
        phrasing = [
            f"Training efforts in {when} resulted in {total} individuals being trained. Top-performing colleges include {trainers}.",
            f"A total of {total} people were trained in {when}, with contributions from {trainers}.",
            f"In {when}, {total} individuals benefited from trainings. Leading colleges include {trainers}.",
        ]
        summary = random.choice(phrasing)

    elif func_name == "get_client_requests":
        total = sum(data_dict.values())
        statuses = ", ".join([f"{k} ({v})" for k, v in data_dict.items()])
        phrasing = [
            f"In {when}, the system handled {total} client requests, primarily consisting of {statuses}.",
            f"{total} client requests were logged in {when}, including {statuses}.",
            f"{when} saw {total} requests, with breakdowns as follows: {statuses}.",
        ]
        summary = random.choice(phrasing)

    elif func_name == "get_sdg_distribution":
        total = sum(data_dict.values())
        sdgs = ", ".join([f"{k} ({v})" for k, v in data_dict.items()])
        phrasing = [
            f"In {when}, SDG-aligned projects were most associated with {sdgs}, totalling {total} entries.",
            f"{total} SDG-related projects were recorded in {when}, highlighting goals such as {sdgs}.",
            f"The year {when} included {total} projects aligned with SDGs like {sdgs}.",
        ]
        summary = random.choice(phrasing)

    else:
        summary = f"Summary for \"{label}\" not available."

    return f"{original_output.strip()}\n\n{summary.strip()}\n"


######################################################################################################################################################


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


######################################################################################################################################################


# --- AI Summary Entry Point ---
def ai_summary(field_of_focus, timeline_type, value):
    func = detect_function(field_of_focus)
    year, month = parse_timeline(timeline_type, value)
    if not year:
        return "⚠️ Invalid timeline input."

    filters = extract_filters(field_of_focus)  # Extract once

    results = dispatch(func, year, month, filters)  # Pass filters
    return summarize_output(results, f"{field_of_focus} in {value}", filters, func)  # Use filters here too


# === Example Usage ===
if __name__ == "__main__":
    print(ai_summary("Extension Projects", "year", "2024"))
    print(ai_summary("Ongoing Extension Projects", "year", "2024"))
    print(ai_summary("Projects in College of Engineering Architecture and Technology", "year", "2024"))
    print(ai_summary("Budget Distribution", "year", "2022"))
    print(ai_summary("Budget of College of Sciences", "year", "2022"))
    print(ai_summary("Distribution of Agenda Types", "year", "2024"))
    print(ai_summary("Trained Individuals", "year", "2022"))
    print(ai_summary("Trained Individuals in College of Nursing and Health Sciences", "month", "March 2022"))
    print(ai_summary("Client Request", "year", "2020"))
    print(ai_summary("Rejected Client Requests", "month", "April 2020"))
    print(ai_summary("SDG Project Breakdown", "year", "2023"))
