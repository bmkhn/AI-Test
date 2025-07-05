import sqlite3
import random
from faker import Faker

fake = Faker()

# Connect to your SQLite DB
conn = sqlite3.connect("mini_ai_teamgen.db")
cursor = conn.cursor()

# Optional: Clear Previous Data and Reset AUTOINCREMENT Counters
cursor.execute("PRAGMA foreign_keys = ON")
tables = [
    "evaluations",
    "project_teams",
    "projects",
    "users",
    "colleges"
]
for table in tables:
    cursor.execute(f"DELETE FROM {table}")
    try:
        cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
    except sqlite3.OperationalError:
        pass  # Skip if sqlite_sequence doesn't exist

# --- 1. Insert Colleges ---
college_names = ["College of Arts and Humanities", "College of Business and Accountancy", "College of Criminal Justice Education", "College of Engineering Architecture and Technology", "College of Hospitality Management and Tourism", "College of Nursing and Health Sciences", "College of Sciences", "College of Teacher Education"]
cursor.executemany("INSERT INTO colleges (name) VALUES (?)", [(name,) for name in college_names])
conn.commit()

# Fetch inserted colleges with IDs
cursor.execute("SELECT id FROM colleges")
college_ids = [row[0] for row in cursor.fetchall()]

# --- 2. Insert Users ---
campuses = ["PCAT-CUYO", "Tinuigiban", "Araceli", "Bataraza", "Brooke's Point", "Coron", "Dumaran", "Linapacan", "Narra", "Quezon", "Rizal", "San Rafael", "San Vicente", "Sofronio Española", "Taytay"]
degrees = [
    "BA in Communication", "BA in Political Science", "BA in Philippine Studies", "BA in Social Work", "BS in Psychology",                                  # College of Arts and Humanities
    "BS in Accountancy", "BS in Management Accounting", "BS in Business Administration", "BS in Entrepreneurship",                                          # College of Business and Accountancy
    "BS in Criminology",                                                                                                                                    # College of Criminal Justice Education
    "BS in Architecture", "BS in Civil Engineering", "BS in Electrical Engineering", "BS in Mechanical Engineering", "BS in Petroleum Engineering",         # College of Engineering Architecture and Technology
    "BS in Hospitality Management", "BS in Tourism Management",                                                                                             # College of Hospitality Management and Tourism
    "BS in Nursing", "BS in Midwifery",                                                                                                                     # College of Nursing and Health Sciences
    "BS in Biology", "BS in Marine Biology", "BS in Computer Science", "BS in Environmental Science", "BS in Information Technology",                       # College of Sciences
    "B in Elementary Education", "B in Secondary Education", "B in Physical Education"                                                                      # College of Teacher Education
]

degree_expertise_map = {
    # Arts and Humanities
    "BA in Communication": ["Media Production", "Public Relations", "Creative Writing"],
    "BA in Political Science": ["Political Analysis", "Social Advocacy", "Community Development"],
    "BA in Philippine Studies": ["Creative Writing", "Social Advocacy", "Cultural Research"],
    "BA in Social Work": ["Community Development", "Counseling"],
    "BS in Psychology": ["Counseling", "Health Education"],

    # Business
    "BS in Accountancy": ["Accounting", "Taxation", "Auditing"],
    "BS in Management Accounting": ["Financial Reporting", "Business Planning"],
    "BS in Business Administration": ["Entrepreneurship", "Marketing Strategy", "Human Resource Management"],
    "BS in Entrepreneurship": ["Entrepreneurship", "Business Planning", "Marketing Strategy"],

    # Criminology
    "BS in Criminology": ["Criminal Investigation", "Law Enforcement", "Security Management"],

    # Engineering, Architecture, Technology
    "BS in Architecture": ["Architectural Design", "AutoCAD", "Project Estimation"],
    "BS in Civil Engineering": ["Structural Engineering", "Project Estimation"],
    "BS in Electrical Engineering": ["Electrical Systems", "Project Estimation"],
    "BS in Mechanical Engineering": ["Mechanical Design", "Project Estimation"],
    "BS in Petroleum Engineering": ["Petroleum Engineering"],

    # Hospitality
    "BS in Hospitality Management": ["Hotel Management", "Food and Beverage Management"],
    "BS in Tourism Management": ["Travel Consultancy", "Tour Operations"],

    # Health Sciences
    "BS in Nursing": ["Patient Care", "Medical Documentation", "Community Health"],
    "BS in Midwifery": ["Midwifery", "Health Education"],

    # Sciences
    "BS in Biology": ["Biological Research"],
    "BS in Marine Biology": ["Marine Conservation", "Environmental Impact Assessment"],
    "BS in Computer Science": ["Software Development", "Cybersecurity", "Machine Learning"],
    "BS in Environmental Science": ["Environmental Impact Assessment", "Biotechnology"],
    "BS in Information Technology": ["IT Support", "Web Development", "Cloud Computing"],

    # Education
    "B in Elementary Education": ["Curriculum Development", "Lesson Planning"],
    "B in Secondary Education": ["Educational Assessment", "Lesson Planning"],
    "B in Physical Education": ["Physical Education Instruction", "Classroom Management"]
}

users = []
for _ in range(500):
    name = fake.name()
    campus = random.choice(campuses)
    college_id = random.choice(college_ids)
    degree = random.choice(degrees)
    expertise_choices = degree_expertise_map.get(degree, ["Research", "Project Management"])  # Default expertise if degree not found
    expertise = random.choice(expertise_choices)
    users.append((name, campus, college_id, degree, expertise))

cursor.executemany("""
INSERT INTO users (name, campus, college_id, degree, expertise)
VALUES (?, ?, ?, ?, ?)""", users)
conn.commit()

# Get user IDs
cursor.execute("SELECT id FROM users")
user_ids = [row[0] for row in cursor.fetchall()]

# --- 3. Insert Projects ---
statuses = ["Pending", "Ongoing", "Completed"]
projects = []
for i in range(10):
    title = f"Project {fake.word().capitalize()} {i}"
    status = random.choice(statuses)
    projects.append((title, status))

cursor.executemany("INSERT INTO projects (title, status) VALUES (?, ?)", projects)
conn.commit()

# Get project IDs
cursor.execute("SELECT id FROM projects")
project_ids = [row[0] for row in cursor.fetchall()]

# --- 4. Insert Project Teams ---
project_teams = []
for pid in project_ids:
    team_size = random.randint(3, 6)
    team_members = random.sample(user_ids, team_size)
    for uid in team_members:
        project_teams.append((uid, pid))
cursor.executemany("INSERT INTO project_teams (user_id, project_id) VALUES (?, ?)", project_teams)
conn.commit()

# --- 5. Insert Evaluations ---
evaluations = []
for pid in project_ids:
    for _ in range(random.randint(3, 5)):
        rating = random.randint(1, 5)
        comment = fake.sentence()
        evaluations.append((rating, comment, pid))
cursor.executemany("""
INSERT INTO evaluations (rating, comment, project_id)
VALUES (?, ?, ?)""", evaluations)
conn.commit()

print("✅ Populated AI Team Generator Mini DB with Fake Data.")
conn.close()
