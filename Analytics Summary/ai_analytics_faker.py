import sqlite3
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
conn = sqlite3.connect("mini_ai_sumgen.db")
cursor = conn.cursor()

# Optional: Clear Previous Data
cursor.execute("PRAGMA foreign_keys = ON")
tables = [
    "submission_responses",
    "submission_requirements",
    "project_team",
    "sdg_project",
    "projects",
    "client_requests",
    "users",
    "agenda_college",
    "college_budgets",
    "agendas",
    "sdgs",
    "colleges"
]
for table in tables:
    cursor.execute(f"DELETE FROM {table}")
    try:
        cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
    except sqlite3.OperationalError:
        pass  # Skip if sqlite_sequence doesn't exist


def random_date(start_days_ago=1000, end_days_ahead=100):
    start = datetime.now() - timedelta(days=random.randint(0, start_days_ago))
    end = start + timedelta(days=random.randint(1, end_days_ahead))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# --- Colleges ---
colleges = [
    "College of Arts and Humanities", "College of Business and Accountancy",
    "College of Criminal Justice Education", "College of Engineering Architecture and Technology",
    "College of Hospitality Management and Tourism", "College of Nursing and Health Sciences",
    "College of Sciences", "College of Teacher Education"
]
cursor.executemany("INSERT INTO colleges (name) VALUES (?)", [(c,) for c in colleges])
college_ids = [row[0] for row in cursor.execute("SELECT id FROM colleges").fetchall()]


# --- Agendas ---
agenda_placeholders = [(f"Agenda {i+1}", fake.sentence()) for i in range(5)]
cursor.executemany("INSERT INTO agendas (name, description) VALUES (?, ?)", agenda_placeholders)
agenda_ids = [row[0] for row in cursor.execute("SELECT id FROM agendas").fetchall()]


# --- Agenda-College Links ---
agenda_college_pairs = set()
while len(agenda_college_pairs) < 10:
    pair = (random.choice(agenda_ids), random.choice(college_ids))
    agenda_college_pairs.add(pair)

cursor.executemany("INSERT INTO agenda_college (agenda_id, college_id) VALUES (?, ?)",
                list(agenda_college_pairs))



# --- College Budgets ---
cursor.executemany("INSERT INTO college_budgets (college_id, allocated_budget, year) VALUES (?, ?, ?)", [
    (cid, random.randint(1_000_000, 10_000_000), year)
    for cid in college_ids for year in range(2022, 2026)
])


# --- SDGs ---
sdgs = [
    (1, "No Poverty", "End poverty in all its forms everywhere."),
    (2, "Zero Hunger", "End hunger, achieve food security and improved nutrition, and promote sustainable agriculture."),
    (3, "Good Health and Well-being", "Ensure healthy lives and promote well-being for all at all ages."),
    (4, "Quality Education", "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."),
    (5, "Gender Equality", "Achieve gender equality and empower all women and girls."),
    (6, "Clean Water and Sanitation", "Ensure availability and sustainable management of water and sanitation for all."),
    (7, "Affordable and Clean Energy", "Ensure access to affordable, reliable, sustainable and modern energy for all."),
    (8, "Decent Work and Economic Growth", "Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all."),
    (9, "Industry, Innovation and Infrastructure", "Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation."),
    (10, "Reduced Inequality", "Reduce inequality within and among countries."),
    (11, "Sustainable Cities and Communities", "Make cities and human settlements inclusive, safe, resilient and sustainable."),
    (12, "Responsible Consumption and Production", "Ensure sustainable consumption and production patterns."),
    (13, "Climate Action", "Take urgent action to combat climate change and its impacts."),
    (14, "Life Below Water", "Conserve and sustainably use the oceans, seas and marine resources for sustainable development."),
    (15, "Life on Land", "Protect, restore and promote sustainable use of terrestrial ecosystems..."),
    (16, "Peace, Justice and Strong Institutions", "Promote peaceful and inclusive societies..."),
    (17, "Partnerships for the Goals", "Strengthen the means of implementation and revitalize the Global Partnership...")
]
cursor.executemany("INSERT INTO sdgs (goal_number, title, description) VALUES (?, ?, ?)", sdgs)
sdg_ids = [row[0] for row in cursor.execute("SELECT id FROM sdgs").fetchall()]


# --- Users ---
campuses = [
    "PCAT-CUYO", "Tinuigiban", "Araceli", "Bataraza", "Brooke's Point", "Coron",
    "Dumaran", "Linapacan", "Narra", "Quezon", "Rizal", "San Rafael",
    "San Vicente", "Sofronio Española", "Taytay"
]
users = [(fake.name(), random.choice(campuses), random.choice(college_ids)) for _ in range(20)]
cursor.executemany("INSERT INTO users (name, campus, college_id) VALUES (?, ?, ?)", users)
user_ids = [row[0] for row in cursor.execute("SELECT id FROM users").fetchall()]


# --- Projects ---
projects = []
for _ in range(10):
    title = fake.bs().capitalize()
    agenda_id = random.choice(agenda_ids)
    ptype = random.choice(["Research", "Extension"])
    budget = random.randint(100_000, 5_000_000)
    start, end = random_date()
    status = random.choice(["Ongoing", "Completed", "Pending"])
    created = fake.date_this_decade().strftime("%Y-%m-%d")
    projects.append((title, agenda_id, ptype, budget, start, end, status, created))
cursor.executemany("""INSERT INTO projects (title, agenda_id, type, allocated_budget, start_date, end_date, status, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", projects)
project_ids = [row[0] for row in cursor.execute("SELECT id FROM projects").fetchall()]


# --- SDG-Project Links ---
cursor.executemany("INSERT OR IGNORE INTO sdg_project (project_id, sdgs_id) VALUES (?, ?)", [
    (pid, random.choice(sdg_ids)) for pid in project_ids for _ in range(random.randint(1, 3))
])


# --- Project Teams ---
cursor.executemany("INSERT OR IGNORE INTO project_team (project_id, user_id, team_role) VALUES (?, ?, ?)", [
    (pid, uid, random.choice(["Leader", "Member", "Assistant"]))
    for pid in project_ids for uid in random.sample(user_ids, random.randint(2, 5))
])


# --- Submission Requirements ---
subreqs = []
for pid in project_ids:
    for _ in range(random.randint(1, 2)):
        deadline = fake.future_date().strftime("%Y-%m-%d")
        notes = fake.sentence()
        status = random.choice(["Pending", "Submitted", "Late"])
        subreqs.append((pid, deadline, notes, status))
cursor.executemany("INSERT INTO submission_requirements (project_id, deadline, notes, status) VALUES (?, ?, ?, ?)", subreqs)
subreq_ids = [row[0] for row in cursor.execute("SELECT id FROM submission_requirements").fetchall()]


# --- Submission Responses ---
cursor.executemany("""INSERT INTO submission_responses 
    (submitted_file, sub_req_id, trained_individuals, for_product_production, for_research, for_extension, submitted_at) 
    VALUES (?, ?, ?, ?, ?, ?, ?)""", [
    (b'dummy', sid, random.randint(0, 100),
     random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
     fake.date_this_decade().strftime("%Y-%m-%d"))
    for sid in subreq_ids
])


# --- Client Requests ---
cursor.executemany("INSERT INTO client_requests (name, organization, status, created_at) VALUES (?, ?, ?, ?)", [
    (fake.catch_phrase(), fake.company(), random.choice(["New", "Accepted", "Rejected"]),
     fake.date_time_this_decade().strftime("%Y-%m-%d"))
    for _ in range(5)
])

conn.commit()
conn.close()
print("✅ Populated Analytics AI Summary Mini DB with Fake Data.")
