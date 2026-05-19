from flask import Flask, request, jsonify, send_from_directory
import threading
from flask_cors import CORS
import json
import re
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configure logging
logging.basicConfig(
    filename='chatbot_queries.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables
knowledge_base = {}
dynamic_json_data = {}
placement_data = {}
session_context = {}
kb_embeddings = None
kb_keys = []
file_lock = threading.Lock()
MAX_SESSIONS = 1000

# Greetings and farewells with emojis
greetings = {
    "hi": "Hello! Welcome to the RVR & JC College Chatbot. How can I assist you today? 😊🏫",
    "hello": "Hey there! Excited to help you explore RVR & JC College. What's on your mind? 🎉",
    "hey": "Hi! Ready to answer your questions about RVR & JC. What's up? 🚀"
}

farewells = {
    "bye": "Goodbye! Feel free to return anytime for more info on RVR & JC! 👋",
    "goodbye": "See you later! Stay curious about RVR & JC! 🌟",
    "see you": "Catch you next time! Happy exploring! 😄"
}

# Keywords for query classification
keywords = {
    "history": ["history", "founded", "established"],
    "safety": ["safety", "security", "safe"],
    "programs": ["program", "programs", "course", "courses", "btech", "mtech", "mba", "mca"],
    "admission": ["admission", "admissions", "process", "apply"],
    "transportation": ["transport", "transportation", "bus", "buses"],
    "placements": ["placement", "placements", "job", "jobs", "recruitment", "lacement"],
    "placement_years": ["2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    "placement_departments": ["cse", "csd", "csbs", "ece", "eee", "it", "mechanical", "civil", "chemical", "mca", "mba"],
    "placement_head": ["highest package", "highest salary", "top package", "lowest package", "lowest salary", "average package"],
    "placement_cell_head": ["head of placement cell", "placement cell head", "head of the lacement cell"],
    "hostel": ["hostel", "hostels", "accommodation", "residence", "dormitory", "boys hostel", "girls hostel", "mens hostel", "womens hostel"],
    "extracurricular": ["extracurricular", "club", "clubs", "activities", "events"],
    "sports": ["sport", "sports", "cricket", "badminton", "table tennis", "gym", "indoor games", "swimming"],
    "campus": ["campus", "facility", "facilities", "infrastructure", "timings", "college"],
    "location": ["location", "address", "where"],
    "events": ["event", "events", "fest", "fests", "festival"],
    "departments": ["department", "departments", "cse", "csd", "csbs", "ece", "eee", "it", "mechanical", "civil", "chemical", "mca", "mba"],
    "hod": ["hod", "head of department"],
    "principal": ["principal"],
    "staff": ["staff", "faculty", "teachers"],
    "size": ["size", "area", "acres"],
    "blocks": ["block", "blocks", "building", "buildings"],
    "fees": ["fee", "fees", "tuition", "cost", "hostel fee", "transportation fee", "management fee"],
    "library": ["library", "books", "timings", "charges", "penalty", "about library"],
    "scholarships": ["scholarship", "scholarships", "financial aid"],
    "research": ["research", "projects", "labs"],
    "alumni": ["alumni", "alumnus", "graduates"],
    "companies_visited": ["companies visited", "recruiters", "top recruiters"],
    "food": ["food", "mess", "canteen", "menu", "store", "snacks", "meals", "items"]
}

def load_knowledge_base():
    global knowledge_base, dynamic_json_data, placement_data, kb_embeddings, kb_keys
    try:
        # Load static data
        with open('RVRJC_static.json', 'r', encoding='utf-8') as f:
            static_data = json.load(f)
        for key, value in static_data.items():
            if isinstance(value, list):
                knowledge_base[key] = "; ".join(value)
            else:
                knowledge_base[key] = value
        logging.debug(f"Static data keys: {list(knowledge_base.keys())}")
        
        # Load dynamic data
        try:
            with open('RVRJC_dynamic.json', 'r', encoding='utf-8') as f:
                dynamic_json_data.update(json.load(f))
            logging.debug(f"Dynamic data loaded: {list(dynamic_json_data.keys())}")
        except FileNotFoundError:
            logging.error("RVRJC_dynamic.json not found")
        
        # Load placement data
        try:
            with open('RVRJC_Placement_Data.json', 'r', encoding='utf-8') as f:
                placement_data.update(json.load(f))
            logging.debug(f"Placement data loaded: {list(placement_data.keys())}")
        except FileNotFoundError:
            logging.error("RVRJC_Placement_Data.json not found")
        
        kb_keys = list(knowledge_base.keys())
        if kb_keys:
            kb_embeddings = model.encode(kb_keys, convert_to_tensor=True)
            logging.info("Knowledge base embeddings created")
    except Exception as e:
        logging.error(f"Error loading knowledge base: {str(e)}")

def fetch_dynamic_data():
    try:
        url = "https://rvrjc.ac.in/"
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        announcements = soup.find_all('div', class_='announcement')
        for idx, ann in enumerate(announcements[:3], 1):
            text = ann.get_text(strip=True)
            knowledge_base[f"announcement_{idx}"] = text
            dynamic_json_data[f"announcement_{idx}"] = text
        with file_lock:
            with open('RVRJC_dynamic.json', 'w', encoding='utf-8') as f:
                json.dump(dynamic_json_data, f, ensure_ascii=False, indent=2)
        global kb_embeddings, kb_keys
        kb_keys = list(knowledge_base.keys())
        if kb_keys:
            kb_embeddings = model.encode(kb_keys, convert_to_tensor=True)
        logging.info("Dynamic data and embeddings updated")
    except Exception as e:
        logging.error(f"Error fetching dynamic data: {str(e)}")

def update_dynamic_data():
    if not os.path.exists('RVRJC_dynamic.json') or (datetime.now() - datetime.fromtimestamp(os.path.getmtime('RVRJC_dynamic.json'))).total_seconds() > 3600:
        fetch_dynamic_data()

def handle_placement_query(query_clean, matched_category, year, department):
    response = None
    placements = placement_data.get("Placements", {})
    logging.debug(f"Handling placement query: category={matched_category}, year={year}, department={department}, query_clean={query_clean}")

    dept_mapping = {
        "cse": "Computer Science and Engineering",
        "csd": "Computer Science and Data Science",
        "csbs": "Computer Science and Machine Learning",
        "ece": "Electronics and Communication Engineering",
        "eee": "Electrical and Electronics Engineering",
        "it": "Information Technology",
        "mechanical": "Mechanical Engineering",
        "civil": "Civil Engineering",
        "chemical": "Chemical Engineering",
        "mca": "MCA",
        "mba": "Management Sciences"
    }

    if matched_category == "placement_head":
        package_type = "highest"
        if "lowest" in query_clean:
            package_type = "lowest"
        elif "average" in query_clean:
            package_type = "average"
        
        package_key = {
            "highest": "3. Highest Package",
            "lowest": "5. Lowest Package",
            "average": "4. Average Package"
        }.get(package_type, "3. Highest Package")
        
        if year:
            package = placements.get(package_key, {}).get(year, "Not specified")
            logging.debug(f"Year-specific package lookup: key={package_key}, year={year}, package={package}")
            response = f"The {package_type} package in {year} was {package}. Curious about department-wise placements or top recruiters? 💼"
        else:
            try:
                highest_data = placements.get("3. Highest Package", {})
                lowest_data = placements.get("5. Lowest Package", {})
                average_data = placements.get("4. Average Package", {})
                
                highest_package = "Not specified"
                highest_year = ""
                max_value = 0
                for y, p in highest_data.items():
                    if p != "Not specified":
                        try:
                            value = float(p.replace(" LPA", ""))
                            if value > max_value:
                                max_value = value
                                highest_package = p
                                highest_year = y
                        except (ValueError, AttributeError):
                            continue
                
                lowest_package = "Not specified"
                lowest_year = ""
                min_value = float('inf')
                for y, p in lowest_data.items():
                    if p != "Not specified":
                        try:
                            value = float(p.replace(" LPA", ""))
                            if value < min_value:
                                min_value = value
                                lowest_package = p
                                lowest_year = y
                        except (ValueError, AttributeError):
                            continue
                
                average_package = "Not specified"
                average_year = ""
                if average_data:
                    valid_years = [y for y, p in average_data.items() if p != "Not specified"]
                    if valid_years:
                        latest_year = max(valid_years)
                        average_package = average_data.get(latest_year, "Not specified")
                        average_year = latest_year
                
                if package_type == "highest":
                    response = f"Highest package: {highest_package} in {highest_year}, Lowest package: {lowest_package} in {lowest_year}, Average package: {average_package} in {average_year}. Curious about department-wise placements or top recruiters? 💼"
                elif package_type == "lowest":
                    response = f"The lowest package was {lowest_package} in {lowest_year}. Want to know about highest or average packages? 💼"
                elif package_type == "average":
                    response = f"The average package was {average_package} in {average_year}. Want to know about highest or lowest packages? 💼"
                logging.debug(f"Overall package response: {response}")
            except (KeyError, ValueError):
                response = f"No {package_type} package data available. 📊"
    
    elif matched_category == "placement_cell_head":
        head_data = placements.get("11. Placement Head Name", {})
        head = head_data.get("Name", "Not specified")
        response = f"{head} is the head of the placement cell at RVR & JC College, overseeing a robust placement process with top recruiters like TCS, Cognizant, and Wipro. 💼"
        logging.debug(f"Placement cell head response: {head}")
    
    elif matched_category == "companies_visited":
        if year:
            stats = placements.get("1. Year-wise Placement Statistics", {}).get(year, {})
            companies = stats.get("Companies Visited", "Not specified")
            key_recruiters = ", ".join(stats.get("Key Recruiters", {}).keys())
            response = f"In {year}, {companies} companies visited, including key recruiters like {key_recruiters}. Want to know about specific years or departments? 💼"
            logging.debug(f"Companies visited response: year={year}, companies={companies}, recruiters={key_recruiters}")
        else:
            year_stats = placements.get("1. Year-wise Placement Statistics", {})
            if year_stats:
                latest_year = max(year_stats.keys())
                stats = year_stats.get(latest_year, {})
                companies = stats.get("Companies Visited", "Not specified")
                key_recruiters = ", ".join(stats.get("Key Recruiters", {}).keys())
                response = f"In {latest_year}, {companies} companies visited, including key recruiters like {key_recruiters}. Want to know about specific years or departments? 💼"
                logging.debug(f"Companies visited response: latest_year={latest_year}, companies={companies}, recruiters={key_recruiters}")
            else:
                response = "No data on companies visited available. 📊"
    
    else:
        if "head" in query_clean and "placement" in query_clean:
            head_data = placements.get("11. Placement Head Name", {})
            head = head_data.get("Name", "Not specified")
            response = f"{head} is the head of the placement cell at RVR & JC College, overseeing a robust placement process with top recruiters like TCS, Cognizant, and Wipro. 💼"
            logging.debug(f"Fallback placement cell head response: {head}")
        elif year and department:
            dept_name = dept_mapping.get(department.lower(), department)
            dept_stats = placements.get("6. Department-wise Placements", {}).get(year, {}).get(dept_name, "Not specified")
            year_stats = placements.get("1. Year-wise Placement Statistics", {}).get(year, {})
            response = f"In {year}, {dept_name} had a {dept_stats} placement rate. Companies visited: {year_stats.get('Companies Visited', 'N/A')}, Key recruiters: {', '.join(year_stats.get('Key Recruiters', {}).keys())}. 💼"
            logging.debug(f"Dept-year placement response: dept={dept_name}, year={year}, stats={dept_stats}")
        elif year:
            stats = placements.get("1. Year-wise Placement Statistics", {}).get(year, {})
            response = f"In {year}, {stats.get('Companies Visited', 'N/A')} companies visited, offering roles with recruiters like {', '.join(stats.get('Key Recruiters', {}).keys())}. Placement percentage: {placements.get('2. Placement Percentage', {}).get(year, 'N/A')}. 💼"
            logging.debug(f"Year placement response: year={year}, stats={stats}")
        elif department:
            dept_name = dept_mapping.get(department.lower(), department)
            latest_year = max(placements.get("6. Department-wise Placements", {}).keys())
            dept_stats = placements.get("6. Department-wise Placements", {}).get(latest_year, {}).get(dept_name, "Not specified")
            response = f"In {latest_year}, {dept_name} had a {dept_stats} placement rate. Curious about specific years or other departments? 💼"
            logging.debug(f"Dept placement response: dept={dept_name}, latest_year={latest_year}, stats={dept_stats}")
        else:
            latest_year = max(placements.get("1. Year-wise Placement Statistics", {}).keys())
            stats = placements.get("1. Year-wise Placement Statistics", {}).get(latest_year, {})
            response = f"In {latest_year}, {stats.get('Companies Visited', 'N/A')} companies visited, offering roles with recruiters like {', '.join(stats.get('Key Recruiters', {}).keys())}. Placement percentage: {placements.get('2. Placement Percentage', {}).get(latest_year, 'N/A')}. 💼"
            logging.debug(f"Overall placement response: latest_year={latest_year}, stats={stats}")

    if not response or "not specified" in response.lower() or "no " in response.lower():
        query = f"What is the {package_type if matched_category == 'placement_head' else 'overall placement statistics'} at RVR & JC College"
        if matched_category == "placement_cell_head" or ("head" in query_clean and "placement" in query_clean):
            query = "Who is the head of the placement cell at RVR & JC College"
        if matched_category == "companies_visited":
            query = "What are the companies visited at RVR & JC College"
        if year:
            query = f"What are the placement statistics at RVR & JC College for {year}"
        if department:
            query = f"What are the placement statistics for {department} at RVR & JC College"
        response = knowledge_base.get(query, f"I couldn’t find specific placement details, but RVR & JC has a strong placement record! Try asking about a specific year or department. 📊")
        logging.debug(f"Fallback to knowledge_base: query={query}, response={response}")

    return response


def map_query(query_clean, matched_category):
    query = query_clean
    if matched_category == "food" or any(k in query_clean for k in ["food", "canteen", "mess", "store"]):
        if "canteen" in query_clean and ("location" in query_clean or "where" in query_clean):
            query = "Where is the canteen located at RVR & JC College"
        elif "store" in query_clean and ("location" in query_clean or "where" in query_clean):
            query = "Where is the campus store located at RVR & JC College"
        elif "canteen" in query_clean and ("food" in query_clean or "items" in query_clean or "menu" in query_clean):
            query = "What food items are available in the canteen at RVR & JC College"
        elif "store" in query_clean and ("food" in query_clean or "items" in query_clean or "menu" in query_clean):
            query = "What food items are available in the campus store at RVR & JC College"
        elif "canteen" in query_clean:
            query = "What are the canteen facilities at RVR & JC College"
        elif "store" in query_clean or "shop" in query_clean:
            query = "What is the campus store at RVR & JC College"
        elif "hostel" in query_clean or "mess" in query_clean or "menu" in query_clean:
            query = "What types of food are provided in the hostels at RVR & JC College"
        else:
            query = "What are the canteen facilities at RVR & JC College"
    elif matched_category == "library" or "library" in query_clean:
        if "location" in query_clean or "where" in query_clean or "address" in query_clean:
            query = "Where is the library located at RVR & JC College"
        elif any(k in query_clean for k in ["number", "how many", "volumes", "titles"]):
            query = "What is the number of books in the library at RVR & JC College"
        elif any(k in query_clean for k in ["return", "duration", "how long"]):
            query = "What is the duration to return books to the library at RVR & JC College"
        elif "-timing" in query_clean or "timings" in query_clean or "hours" in query_clean:
            query = "What are the timings of the library at RVR & JC College"
        elif any(k in query_clean for k in ["charges", "penalty", "cost", "fine"]):
            query = "What are the charges or penalties for the library at RVR & JC College"
        elif any(k in query_clean for k in ["types", "type", "journals"]) and not any(k in query_clean for k in ["return", "duration"]):
            query = "What types of books are available in the library at RVR & JC College"
        else:
            query = "What are the library facilities at RVR & JC College"
    elif matched_category == "hostel" or "hostel" in query_clean:
        if "availability" in query_clean or "available" in query_clean or ("boys" in query_clean and "girls" in query_clean):
            query = "What is the availability of hostels for boys and girls at RVR & JC College"
        elif "location" in query_clean or "where" in query_clean or "address" in query_clean:
            if "boys" in query_clean or "boys hostel" in query_clean:
                query = "Where is the boys' hostel located at RVR & JC College"
            elif any(k in query_clean for k in ["girls", "girl", "girls'", "girl's", "womens", "women's hostel"]):
                query = "Where is the girls' hostel located at RVR & JC College"
            else:
                query = "What are the hostel facilities at RVR & JC College"
        elif "fees" in query_clean or "cost" in query_clean or "price" in query_clean or "hostel fee" in query_clean:
            if "boys" in query_clean or "hostel fee for boys" in query_clean:
                query = "What is the hostel fee for boys at RVR & JC College"
            elif any(k in query_clean for k in ["girls", "girl", "girls'", "girl's", "womens", "women's hostel", "hostel fee for girls"]):
                query = "What is the hostel fee for girls at RVR & JC College"
            else:
                query = "What are the hostel fees at RVR & JC College"
        elif "timing" in query_clean or "timings" in query_clean or "schedule" in query_clean:
            query = "What are the hostel timings at RVR & JC College"
        elif "food" in query_clean or "mess" in query_clean or "menu" in query_clean:
            query = "What types of food are provided in the hostels at RVR & JC College"
        elif "rules" in query_clean or "regulation" in query_clean or "discipline" in query_clean:
            query = "What are the hostel rules at RVR & JC College"
        elif "facility" in query_clean or "facilities" in query_clean:
            query = "What are the hostel facilities at RVR & JC College"
        else:
            query = "What are the hostel facilities at RVR & JC College"
    elif matched_category == "fees" or "fees" in query_clean:
        if "department-wise tuition fee" in query_clean or "management fee" in query_clean or any(k in query_clean for k in ["btech fee", "mtech fee", "mca fee", "mba fee", "department wise fee"]):
            query = "What is the department-wise tuition fee and management fee at RVR & JC College"
        elif "hostel fee for boys" in query_clean:
            query = "What is the hostel fee for boys at RVR & JC College"
        elif "hostel fee for girls" in query_clean:
            query = "What is the hostel fee for girls at RVR & JC College"
        elif "transportation fees" in query_clean:
            query = "What are the transportation fees at RVR & JC College"
        elif "hostel fees" in query_clean:
            query = "What are the hostel fees at RVR & JC College"
        else:
            query = "What is the fee structure for B.Tech programs at RVR & JC College"
    elif matched_category == "departments" or matched_category == "hod":
        if "hod" in query_clean or "head of department" in query_clean:
            if "cse" in query_clean and not ("csd" in query_clean or "data science" in query_clean):
                query = "Who is the HOD of the Computer Science & Engineering department at RVR & JC College"
            elif "csd" in query_clean or "data science" in query_clean:
                query = "Who is the HOD of the Computer Science & Engineering (Data Science) department at RVR & JC College"
            elif "csbs" in query_clean or "machine learning" in query_clean:
                query = "Who is the HOD of the Computer Science and Machine Learning department at RVR & JC College"
            elif "chemical" in query_clean:
                query = "Who is the HOD of the Chemical Engineering department at RVR & JC College"
            elif "civil" in query_clean:
                query = "Who is the HOD of the Civil Engineering department at RVR & JC College"
            elif "ece" in query_clean or "electronics" in query_clean:
                query = "Who is the HOD of the Electronics & Communication Engineering department at RVR & JC College"
            elif "eee" in query_clean or "electrical" in query_clean:
                query = "Who is the HOD of the Electrical & Electronics Engineering department at RVR & JC College"
            elif "it" in query_clean or "information technology" in query_clean:
                query = "Who is the HOD of the Information Technology department at RVR & JC College"
            elif "mechanical" in query_clean:
                query = "Who is the HOD of the Mechanical Engineering department at RVR & JC College"
            elif "mca" in query_clean:
                query = "Who is the HOD of the MCA department at RVR & JC College"
            elif "mba" in query_clean or "management sciences" in query_clean:
                query = "Who is the HOD of the Management Sciences department at RVR & JC College"
            else:
                query = "What are the departments at RVR & JC College"
        elif "lab" in query_clean or "laboratory" in query_clean:
            if "csd" in query_clean or "data science" in query_clean:
                query = "What are the laboratory facilities in the Computer Science & Engineering (Data Science) department at RVR & JC College"
            elif "cse" in query_clean:
                query = "What are the laboratory facilities in the Computer Science & Engineering department at RVR & JC College"
            elif "csbs" in query_clean or "machine learning" in query_clean:
                query = "What are the laboratory facilities in the Computer Science and Machine Learning department at RVR & JC College"
            elif "chemical" in query_clean:
                query = "What are the laboratory facilities in the Chemical Engineering department at RVR & JC College"
            elif "civil" in query_clean:
                query = "What are the laboratory facilities in the Civil Engineering department at RVR & JC College"
            elif "ece" in query_clean or "electronics" in query_clean:
                query = "What are the laboratory facilities in the Electronics & Communication Engineering department at RVR & JC College"
            elif "eee" in query_clean or "electrical" in query_clean:
                query = "What are the laboratory facilities in the Electrical & Electronics Engineering department at RVR & JC College"
            elif "it" in query_clean or "information technology" in query_clean:
                query = "What are the laboratory facilities in the Information Technology department at RVR & JC College"
            elif "mechanical" in query_clean:
                query = "What are the laboratory facilities in the Mechanical Engineering department at RVR & JC College"
            elif "mca" in query_clean:
                query = "What are the laboratory facilities in the MCA department at RVR & JC College"
            elif "mba" in query_clean or "management sciences" in query_clean:
                query = "What are the laboratory facilities in the Management Sciences department at RVR & JC College"
            else:
                query = "What are the departments at RVR & JC College"
        elif "fees" in query_clean:
            query = "What is the department-wise tuition fee and management fee at RVR & JC College"
        else:
            if "csd" in query_clean or "data science" in query_clean:
                query = "What programs are offered by the Computer Science & Engineering (Data Science) department at RVR & JC College"
            elif "cse" in query_clean:
                query = "What programs are offered by the Computer Science & Engineering department at RVR & JC College"
            elif "csbs" in query_clean or "machine learning" in query_clean:
                query = "What programs are offered by the Computer Science and Machine Learning department at RVR & JC College"
            elif "chemical" in query_clean:
                query = "What programs are offered by the Chemical Engineering department at RVR & JC College"
            elif "civil" in query_clean:
                query = "What programs are offered by the Civil Engineering department at RVR & JC College"
            elif "ece" in query_clean or "electronics" in query_clean:
                query = "What programs are offered by the Electronics & Communication Engineering department at RVR & JC College"
            elif "eee" in query_clean or "electrical" in query_clean:
                query = "What programs are offered by the Electrical & Electronics Engineering department at RVR & JC College"
            elif "it" in query_clean or "information technology" in query_clean:
                query = "What programs are offered by the Information Technology department at RVR & JC College"
            elif "mechanical" in query_clean:
                query = "What programs are offered by the Mechanical Engineering department at RVR & JC College"
            elif "mca" in query_clean:
                query = "What programs are offered by the MCA department at RVR & JC College"
            elif "mba" in query_clean or "management sciences" in query_clean:
                query = "What programs are offered by the Management Sciences department at RVR & JC College"
            else:
                query = "What are the departments at RVR & JC College"
    elif any(k in query_clean for k in ["rvrjc", "college"]) and not any(k in query_clean for k in keywords["sports"] + keywords["campus"] + keywords["departments"] + keywords["location"] + keywords["events"] + keywords["transportation"] + keywords["history"] + keywords["safety"] + keywords["programs"] + keywords["fees"] + keywords["placements"] + keywords["hostel"] + keywords["extracurricular"] + keywords["scholarships"] + keywords["research"] + keywords["alumni"] + keywords["principal"] + keywords["staff"] + keywords["size"] + keywords["blocks"] + keywords["library"] + keywords["food"]):
        query = "What is the history of RVR & JC College"
    elif matched_category == "history":
        query = "What is the history of RVR & JC College"
    elif matched_category == "safety":
        query = "What safety measures are in place at RVR & JC College"
    elif matched_category == "programs":
        query = "What are the undergraduate programs offered at RVR & JC College" if "undergraduate" in query_clean else "What are the postgraduate programs offered at RVR & JC College"
    elif matched_category == "admission":
        query = "What is the admission process for RVR & JC College"
    elif matched_category == "extracurricular":
        query = "What extracurricular activities are available at RVR & JC College"
    elif matched_category == "scholarships":
        query = "What scholarships are available at RVR & JC College"
    elif matched_category == "research":
        query = "What are the research opportunities at RVR & JC College"
    elif matched_category == "alumni":
        query = "What is the alumni network like at RVR & JC College"
    elif matched_category == "sports":
        if "swimming" in query_clean:
            query = "Is there a swimming facility at RVR & JC College"
        elif "cricket" in query_clean:
            query = "What are the cricket facilities at RVR & JC College"
        elif "gym" in query_clean:
            query = "What gym facilities are available at RVR & JC College"
        elif "badminton" in query_clean:
            query = "What are the badminton facilities at RVR & JC College"
        elif "table tennis" in query_clean:
            query = "What are the table tennis facilities at RVR & JC College"
        elif "indoor games" in query_clean:
            query = "What indoor games are available at RVR & JC College"
        else:
            query = "What are the sports facilities at RVR & JC College"
    elif matched_category == "campus":
        query = "What are the college facilities at RVR & JC College"
    elif matched_category == "location":
        query = "Where is RVR & JC College located"
    elif matched_category == "events":
        query = "Are any events scheduled this week"
    elif matched_category == "transportation":
        query = "What is the transportation availability for RVR & JC College"
    elif matched_category == "principal":
        query = "Who is the principal of RVR & JC College"
    elif matched_category == "staff":
        query = "What is the staff strength at RVR & JC College"
    elif matched_category == "size":
        query = "What is the size of RVR & JC College"
    elif matched_category == "blocks":
        query = "What are the blocks in RVR & JC College"

    return query

def find_answer(user_message, session_id):
    corrected = user_message.lower().strip()
    logging.debug(f"Original query: {user_message}, Corrected: {corrected}")

    sentiment = analyzer.polarity_scores(user_message)['compound']
    logging.info(f"Sentiment score: {sentiment}")

    if len(session_context) > MAX_SESSIONS:
        session_context.clear()
        logging.info("Cleared all sessions due to exceeding MAX_SESSIONS.")
    if session_id not in session_context:
        session_context[session_id] = []
    session_context[session_id].append(f"User: {corrected}")
    session_context[session_id] = session_context[session_id][-5:]
    context = "\n".join(session_context[session_id])
    logging.debug(f"Session context: {context}")

    query_clean = corrected

    # Greetings
    for greet in greetings:
        if corrected.strip() == greet:
            return greetings[greet] if sentiment <= 0.4 else f"Wow, you're pumped! {greetings[greet]} 🎈"

    # Name detection
    name_match = re.search(r"\b(i am|my name is)\s+([A-Za-z]+)", corrected)
    invalid_names = {"asking", "interested", "looking", "student"}
    if name_match:
        candidate = name_match.group(2).strip().lower()
        if candidate not in invalid_names:
            user_name = candidate.capitalize()
            session_context[session_id].append(f"Name: {user_name}")
            return f"Hello {user_name}! Excited to help you explore RVR & JC! What’s your question? 😊"

    # Farewells
    for farewell in farewells:
        if corrected.strip() == farewell:
            return farewells[farewell] if sentiment <= 0.4 else f"Love your vibe! {farewells[farewell]} ✨"

    # Exact matches for common queries
    exact_matches = {
        "college facilities": "What are the college facilities at RVR & JC College",
        "campus facilities": "What are the college facilities at RVR & JC College",
        "college timings": "What are the timings for students at RVR & JC College",
        "what are the college facilities": "What are the college facilities at RVR & JC College",
        "what are the campus facilities": "What are the college facilities at RVR & JC College",
        "hostel facilities": "What are the hostel facilities at RVR & JC College",
        "what are the hostel facilities": "What are the hostel facilities at RVR & JC College",
        "tuition fees": "What is the department-wise tuition fee and management fee at RVR & JC College",
        "management fees": "What is the department-wise tuition fee and management fee at RVR & JC College",
        "department wise fees": "What is the department-wise tuition fee and management fee at RVR & JC College",
        "boys hostel fees": "What is the hostel fee for boys at RVR & JC College",
        "girls hostel fees": "What is the hostel fee for girls at RVR & JC College",
        "transportation fees": "What are the transportation fees at RVR & JC College",
        "hostel fees": "What are the hostel fees at RVR & JC College",
        "what is the duration to return the books": "What is the duration to return books to the library at RVR & JC College",
        "how long to return library books": "What is the duration to return books to the library at RVR & JC College",
        "what type of books are available in library": "What types of books are available in the library at RVR & JC College",
        "what types of books in library": "What types of books are available in the library at RVR & JC College",
        "sports facilities": "What are the sports facilities at RVR & JC College",
        "what are the sports facilities": "What are the sports facilities at RVR & JC College",
        "who is the head of placement cell": "Who is the head of the placement cell at RVR & JC College",
        "placement cell head": "Who is the head of the placement cell at RVR & JC College",
        "who is the head of the lacement cell": "Who is the head of the placement cell at RVR & JC College",
        "companies visited": "What are the companies visited at RVR & JC College",
        "what about library": "What are the library facilities at RVR & JC College",
        "library": "What are the library facilities at RVR & JC College",
        "canteen facilities": "What are the canteen facilities at RVR & JC College",
        "hostel food": "What types of food are provided in the hostels at RVR & JC College",
        "campus store": "What is the campus store at RVR & JC College",
        "what types of food in hostels": "What types of food are provided in the hostels at RVR & JC College",
        "what is the canteen at rvr & jc": "What are the canteen facilities at RVR & JC College",
        "location of canteen": "Where is the canteen located at RVR & JC College",
        "where is the canteen": "Where is the canteen located at RVR & JC College",
        "location of store": "Where is the campus store located at RVR & JC College",
        "where is the store": "Where is the campus store located at RVR & JC College",
        "food items in canteen": "What food items are available in the canteen at RVR & JC College",
        "canteen food items": "What food items are available in the canteen at RVR & JC College",
        "food items in store": "What food items are available in the campus store at RVR & JC College",
        "store food items": "What food items are available in the campus store at RVR & JC College"
    }

    if corrected in exact_matches:
        query = exact_matches[corrected]
        logging.debug(f"Exact match found: {query}")
        # Direct handling for placement cell head
        if query == "Who is the head of the placement cell at RVR & JC College":
            response = handle_placement_query(query_clean, "placement_cell_head", None, None)
            logging.debug(f"Direct placement cell head response: {response}")
            response = f"{response} 💼"
            return f"Glad you're excited! {response}" if sentiment > 0.4 else response
        elif query in knowledge_base:
            response = knowledge_base[query]
            logging.debug(f"Direct response found for query: {query}")
            # Add category-specific emojis
            if "library" in query.lower():
                response += " 📚 Want to know about library timings, book types, or penalties?"
            elif "hostel" in query.lower():
                response += " 🏠 Want to know about hostel rules, fees, or food?"
            elif "food" in query.lower() or "canteen" in query.lower() or "store" in query.lower():
                response += " 🍽️ Want to know more about hostel mess, canteen items, or the store?"
            elif "sports" in query.lower():
                response += " ⚽ Want to know about specific sports or facilities?"
            elif "fees" in query.lower():
                response += " 💰 Curious about other fees or scholarships?"
            else:
                response += " 🏫 Want to dive into specific details?"
            return f"Glad you're excited! {response}" if sentiment > 0.4 else response
    else:
        # Announcements
        if "latest announcement" in query_clean or "latest events" in query_clean:
            update_dynamic_data()
            latest = [v for k, v in knowledge_base.items() if "announcement_" in k][:3]
            response = "Exciting updates at RVR & JC! 📢\n" + "\n".join(latest) if latest else "No new announcements yet, but stay tuned! 📣"
            response += "\nWant to know about fests or placements?"
            return f"Glad you're excited! {response}" if sentiment > 0.4 else response

        # Query normalization
        query_clean = re.sub(r"\b(please|can i know|tell me about|of)\b", "", query_clean).strip()
        query_clean = re.sub(r"girl's|girls'|girl\b", "girls", query_clean)
        query_clean = re.sub(r"\bpenaulty\b", "penalty", query_clean)
        query_clean = re.sub(r"\blacement\b", "placement", query_clean)
        query_clean = re.sub(r"\b(tuition fee|management fee|department wise fee)\b", "department-wise tuition fee", query_clean)
        query_clean = re.sub(r"\b(boys hostel fee|boys hostel fees)\b", "hostel fee for boys", query_clean)
        query_clean = re.sub(r"\b(girls hostel fee|girls hostel fees|womens hostel fee)\b", "hostel fee for girls", query_clean)
        query_clean = re.sub(r"\b(transport fee|transport fees|transportation fee|transportation fees)\b", "transportation fees", query_clean)
        query_clean = re.sub(r"\b(facilities|Facilities|FACILITIES)\b", "facilities", query_clean)
        query_clean = re.sub(r"\b(faiclities)\b", "facilities", query_clean)
        query_clean = " ".join(query_clean.split())
        logging.debug(f"Cleaned query: {query_clean}")

        # Keyword matching
        matched_category = None
        matched_keywords = []
        for key, variants in keywords.items():
            for v in variants:
                if v in query_clean:
                    matched_keywords.append((key, v))
        
        if any(kw[0] == "placement_cell_head" for kw in matched_keywords):
            matched_category = "placement_cell_head"
        elif any("head" in kw[1] and "placement" in query_clean for kw in matched_keywords):
            matched_category = "placement_cell_head"
        elif any(kw[0] == "food" for kw in matched_keywords):
            matched_category = "food"
        elif ("lab" in query_clean or "laboratory" in query_clean or "hod" in query_clean or "head of department" in query_clean) and any(kw[0] == "departments" for kw in matched_keywords):
            matched_category = "departments"
        elif any(kw[0] == "companies_visited" for kw in matched_keywords):
            matched_category = "companies_visited"
        elif any(kw[0] == "sports" for kw in matched_keywords) and any(kw[1] in keywords["sports"] for kw in matched_keywords):
            matched_category = "sports"
        elif any(kw[0] == "library" for kw in matched_keywords):
            matched_category = "library"
        elif any(kw[0] in ["hostel", "fees"] for kw in matched_keywords):
            for kw in matched_keywords:
                if kw[0] in ["hostel", "fees"]:
                    matched_category = kw[0]
                    break
        else:
            matched_category = matched_keywords[0][0] if matched_keywords else None
        logging.debug(f"Matched keywords: {matched_keywords}, Selected category: {matched_category}")

        # Extract year
        year = None
        for y in keywords["placement_years"]:
            if y in query_clean:
                year = f"{y}-{int(y)+1}"
                break

        # Extract department
        department = None
        for dept in keywords["placement_departments"]:
            if dept in query_clean:
                department = dept
                break

        # Call refactored map function
        query = map_query(query_clean, matched_category)
        logging.debug(f"Final mapped query: {query}")

    # Context follow-ups
    if "i mean" in query_clean and session_context[session_id][-2:]:
        prev_query = session_context[session_id][-2].replace("User: ", "").lower()
        prev_query = re.sub(r"girl's|girls'|girl\b", "girls", prev_query)
        prev_query = re.sub(r"\bpenaulty\b", "penalty", prev_query)
        prev_query = re.sub(r"\blacement\b", "placement", prev_query)
        prev_query = re.sub(r"\b(tuition fee|management fee|department wise fee)\b", "department-wise tuition fee", prev_query)
        prev_query = re.sub(r"\b(boys hostel fee|boys hostel fees)\b", "hostel fee for boys", prev_query)
        prev_query = re.sub(r"\b(girls hostel fee|girls hostel fees|womens hostel fee)\b", "hostel fee for girls", prev_query)
        prev_query = re.sub(r"\b(transport fee|transport fees|transportation fee|transportation fees)\b", "transportation fees", prev_query)
        prev_query = re.sub(r"\b(faiclities)\b", "facilities", prev_query)
        for key, variants in keywords.items():
            if any(k in prev_query for k in variants):
                matched_category = key
                break
        if matched_category == "placement_cell_head" or ("head" in prev_query and "placement" in prev_query):
            response = handle_placement_query(prev_query, "placement_cell_head", None, None)
            response = f"{response} 💼"
            logging.debug(f"Follow-up placement cell head response: {response}")
            return f"Glad you're excited! {response}" if sentiment > 0.4 else response
        elif matched_category in ["placements", "placement_head", "companies_visited"]:
            year = None
            for y in keywords["placement_years"]:
                if y in prev_query:
                    year = f"{y}-{int(y)+1}"
                    break
            department = None
            for dept in keywords["placement_departments"]:
                if dept in prev_query:
                    department = dept
                    break
            response = handle_placement_query(prev_query, matched_category, year, department)
            response = f"{response} 💼"
            logging.debug(f"Follow-up placement response: {response}")
            return f"Glad you're excited! {response}" if sentiment > 0.4 else response
        else:
            query = map_query(prev_query, matched_category)

    # Direct response
    logging.debug(f"Checking query in knowledge_base: {query}")
    if query in knowledge_base:
        response = knowledge_base[query]
        logging.debug(f"Direct response found for query: {query}")
        # Add category-specific emojis
        if "library" in query.lower():
            response += " 📚 Want to know about library timings, book types, or penalties?"
        elif "hostel" in query.lower():
            response += " 🏠 Want to know about hostel rules, fees, or food?"
        elif "placement" in query.lower() or "companies visited" in query.lower():
            response += " 💼 Curious about department-wise placements or top recruiters?"
        elif "facilities" in query.lower():
            response += " 🏫 Want details on specific facilities like the library or sports?"
        elif "fee" in query.lower() or "fees" in query.lower():
            response += " 💰 Curious about other fees or scholarships?"
        elif "food" in query.lower() or "canteen" in query.lower() or "store" in query.lower():
            response += " 🍽️ Want to know more about hostel mess, canteen items, or the store?"
        elif "sports" in query.lower():
            response += " ⚽ Want to know about specific sports or facilities?"
        elif "events" in query.lower() or "fest" in query.lower():
            response += " 🎉 Curious about upcoming fests or campus events?"
        else:
            response += " 🏫 Want to dive into specific placement details or campus life?"
        return f"Glad you're excited! {response}" if sentiment > 0.4 else response
    else:
        logging.warning(f"Query not found in knowledge_base: {query}")

    # Similarity matching
    if kb_keys:
        query_embedding = model.encode(query_clean, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, kb_embeddings)[0]
        top_index = scores.argmax().item()
        top_score = scores[top_index].item()
        logging.debug(f"Similarity top match: {kb_keys[top_index]}, Score: {top_score}")
        if top_score > 0.3:
            matched_key = kb_keys[top_index]
            response = knowledge_base[matched_key]
            logging.debug(f"Similarity response found for key: {matched_key}")
            response += " 🏫 Curious about placements or facilities?"
            return f"Glad you're excited! {response}" if sentiment > 0.4 else response

    # Fallback
    logging.info(f"Unmapped query: {query_clean}, Attempted query: {query}")
    response = f"I couldn’t find specific details for '{query_clean}', but RVR & JC offers a vibrant campus with a central library, modern labs, hostels, sports facilities, and more! Try asking 'What are the college facilities?' or 'What are the sports facilities?' 😊"
    return f"Glad you're excited! {response}" if sentiment > 0.4 else response


@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/clear_session', methods=['POST'])
def clear_session():
    try:
        data = request.json
        session_id = data.get('session_id')
        if session_id and session_id in session_context:
            del session_context[session_id]
            logging.info(f"Cleared session {session_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error clearing session: {str(e)}")
        return jsonify({'status': 'error'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default_session')
        if not user_message:
            return jsonify({'response': 'Please enter a message! 😊'})
        response = find_answer(user_message, session_id)
        if not response:
            response = "I couldn’t process that request, but RVR & JC has a vibrant campus! Try asking about college facilities or sports. 🏫"
        session_context[session_id].append(f"Bot: {response}")
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({'response': f"Oops, something went wrong! But RVR & JC is still awesome—try asking about college facilities or sports. 😊"})

if __name__ == '__main__':
    load_knowledge_base()
    app.run(debug=True)