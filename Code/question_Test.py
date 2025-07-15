import os 
import json
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
labs_folder = os.path.join(parent_folder, "Quizzes")
lab_path = os.path.join(labs_folder, "Quiz2.ube_1.json")
with open(lab_path, "r", encoding="utf-8") as f:
    data = json.load(f)

raw_question = data.get("questions", [])
question_texts = [q.strip().replace("\n", " ") for q in raw_question]
print(question_texts[0])
