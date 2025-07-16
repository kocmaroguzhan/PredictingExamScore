import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast
from transformers import RobertaTokenizer, RobertaModel
import random
import re

# ----------------------------
# Dead Code Injection Utility
# ----------------------------
def generate_dead_code_snippet():
    templates = [
        "int dummy{} = {};",
        'String log{} = "debug";',
        'if (false) {{ System.out.println("debug{}"); }}',
        'boolean flag{} = true;',
        'List<String> list{} = new ArrayList<>();',
        'int temp{} = new Random().nextInt();'
    ]
    template = random.choice(templates)
    count = template.count('{}')
    nums = tuple(random.randint(0, 999) for _ in range(count))
    return template.format(*nums)



# Robust method pattern that supports multiple modifiers
method_pattern = re.compile(
    r'\b(public|private|protected)\b'        # Access modifier
    r'(?:\s+\w+)*'                           # Optional modifiers
    r'\s+[\w<>\[\]]+'                        # Return type
    r'\s+\w+'                                # Method name
    r'\s*\([^)]*\)\s*'                       # Parameter list
)



def inject_dead_code(code: str, min_method_length=6, max_insertions=2) -> str:
    # Split the input Java code into individual lines
    lines = code.strip().split('\n')
    
    # If the code is too short, skip augmentation
    if len(lines) < min_method_length:
        return code, False

    # Create a copy of the lines to modify (original stays untouched)
    dead_code_augmented_lines = lines[:]

    # State variables to track method parsing
    method_start = None         # Line index where current method starts
    method_depth = 0            # Tracks nested block depth using braces
    method_lines = []           # Line indices belonging to the current method
    insert_points = []          # All line indices where dummy code can be inserted
    depth_change_lines = []     
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip comment lines
        if stripped.startswith('//'):
            i += 1
            continue
        # === Method declaration detection ===
        is_method_signature = method_pattern.search(stripped) and method_start is None

        if is_method_signature:
            found_brace = False

            # Case 1: Brace on same line
            if '{' in stripped:
                method_start = i
                method_lines = [i]
                method_depth += stripped.count('{')
                method_depth -= stripped.count('}')
                i += 1
                continue  # Skip to next line
            else:
                # Case 2: Look ahead for brace
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('//') or next_line == '':
                        j += 1
                        continue
                    if '{' in next_line:
                        method_start = j
                        method_lines = [j]
                        method_depth += next_line.count('{')
                        method_depth -= next_line.count('}')
                        i = j + 1  # Skip to line after opening brace
                        found_brace = True
                        break
                    else:
                        break  # Line is not comment/empty and doesn't contain '{' → stop checking

                if not found_brace:
                    # No opening brace found after method signature.
                    # This is likely an abstract method or interface declaration with no body.
                    # We set i = j to continue parsing from the line where the lookahead stopped,
                    # ensuring we don’t reprocess the same lines or skip valid code below.
                    i =j
                continue

        # === Method body tracking ===
        if method_start is not None and method_depth>0:
           # Count braces before update
            prev_depth = method_depth

            # Track nested braces
            method_depth += stripped.count('{')
            method_depth -= stripped.count('}')
            # If depth is decreased, track this line
            if method_depth < prev_depth:
                depth_change_lines.append(i)
            method_lines.append(i)

            # End of method detected
            if method_depth == 0:
                method_len = method_lines[-1] - method_start + 1

                if method_len >= min_method_length:
                    # Exclude last depth change to avoid injecting before lone closing brace (`}`) at method end
                    insert_points.extend(depth_change_lines[:-1])

                # Reset tracking state
                method_start = None
                method_lines = []
                depth_change_lines=[]
        i += 1
 
    if not insert_points:
        return code, False  # No injection possible
    # Randomly select a few insertion points, up to max_insertions
    # Insert from bottom to top to avoid shifting line indices.
    # If we insert at a lower line (e.g., line 5) first, it will push all later lines down by one.
    # By inserting in descending order, previously chosen line numbers remain accurate.
    chosen_insertion_points = sorted(
        random.sample(insert_points, k=min(max_insertions, len(insert_points))),
        reverse=True
    )

    # Insert dummy code before each chosen line
    for idx in chosen_insertion_points:
        # Detect leading whitespace (tabs or spaces)
        indent_match = re.match(r'^(\s*)', dead_code_augmented_lines[idx])
        indent = indent_match.group(1) if indent_match else ''
        # If the line is a closing brace (or ends with one), increase indentation for matching to code block
        if dead_code_augmented_lines[idx].strip() == '}':
            indent += '\t'  # or use '    ' if you prefer spaces
         # Choose a dummy line and append a traceability comment
        dummy_statement = generate_dead_code_snippet()
        dead_code = f"{indent}{dummy_statement} // injected"

        # Insert the formatted dummy line before the actual code line
        dead_code_augmented_lines.insert(idx, dead_code)

    # Return the modified code as a single string
    return "\n".join(dead_code_augmented_lines) ,True

# ----------------------------
# Long Code Embedding Function
# ----------------------------

def get_embedding_for_long_code(code, tokenizer, model, device, chunk_size=512):
    tokenized = tokenizer.encode_plus(
        code,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=False,
        max_length=99999  # large number to avoid truncation here
    )

    input_ids = tokenized['input_ids'].squeeze(0)
    attention_mask = tokenized['attention_mask'].squeeze(0)

    total_tokens = input_ids.size(0)
    stride = chunk_size // 2  # e.g., 256 if chunk_size is 512-- %50 overlap
    embeddings = []

    for start in range(0, total_tokens, stride):
        end = min(start + chunk_size, total_tokens)
        chunk_ids = input_ids[start:end].unsqueeze(0)
        chunk_mask = attention_mask[start:end].unsqueeze(0)

        with torch.no_grad():
            output = model(input_ids=chunk_ids.to(device), attention_mask=chunk_mask.to(device))
            chunk_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy() #CLS token
            embeddings.append(chunk_embedding)

        if end == total_tokens:
            break

    if not embeddings:
        return np.zeros(model.config.hidden_size)

    final_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(final_embedding)
    if norm == 0:
        return final_embedding
    return final_embedding / norm

def get_empty_code_embedding(empty_code_embedding_path):
    # Load the CSV
    df = pd.read_csv(empty_code_embedding_path)

    # Convert string back to list
    empty_code_embedding = ast.literal_eval(df["empty_code_embedding"].iloc[0])

    return empty_code_embedding
# ----------------------------
# Main Embedding Pipeline
# ----------------------------

def generate_all_embeddings(assessment_type:str):
    # assessment_type should be either "quiz" or "lab"
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assessment_folder = os.path.join(parent_folder, "Quizzes" if assessment_type == "quiz" else "Labs")
    dataset_folder = os.path.join(parent_folder, "Dataset")
    augmentation_folder = os.path.join(dataset_folder, "Augmentations")
    empty_code_embedding_path = os.path.join(dataset_folder, "empty_code_embedding.csv")
    empty_code_embedding = get_empty_code_embedding(empty_code_embedding_path)
    os.makedirs(dataset_folder, exist_ok=True)

    config_path = os.path.join(dataset_folder, "config.txt")
    augmentation_count = 0
    number_of_augmented_code_saving_threshold = 0

    with open(config_path, "r") as f:
        for line in f:
            if line.startswith("augmentation_count"):
                augmentation_count = int(line.strip().split("=")[1])
            if line.startswith("number_of_augmented_code_saving_threshold"):
                number_of_augmented_code_saving_threshold = int(line.strip().split("=")[1])


    print("✅ augmentation_count =", augmentation_count)
    print("✅ number_of_augmented_code_saving_threshold =", number_of_augmented_code_saving_threshold)
    
    model_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    student_embeddings = {}
    total_augmentation_count = 0
    last_seen_assessment_section = ""

    for file_name in os.listdir(assessment_folder):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(assessment_folder, file_name)
        assessment_id = os.path.splitext(file_name)[0].split(".")[0]
        assessment_section = os.path.splitext(file_name)[0]##full name of the file
        last_seen_assessment_section = assessment_section

        print(f"📄 Processing {file_name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_augmented_count = 0

        for student in tqdm(data["answers"], desc=f"Embedding {assessment_id}"):
            student_id = student.get("id", "unknown")
            code_sections = [v for k, v in student.items() if k.endswith(".java")]
            full_code = "\n".join(code_sections)

            if saved_augmented_count < 3:
                output_txt_path = os.path.join(augmentation_folder, f"{student_id}_{assessment_id}_original.txt")
                with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(full_code)

            original_embedding = get_embedding_for_long_code(full_code, tokenizer, model, device)

            if student_id not in student_embeddings:
                student_embeddings[student_id] = {}

            if assessment_id not in student_embeddings[student_id]:
                student_embeddings[student_id][assessment_id] = {
                    "original": {},
                    "augmented": [],
                    "assessment_section": assessment_section
                }

            student_embeddings[student_id][assessment_id]["original"] = {
                "embedding": original_embedding.tolist(),
                "valid": True
            }

            code_lines = full_code.strip().split('\n')
            code_length = len(code_lines)
            scale_factor = 0.05
            max_insertions = max(1, int(code_length * scale_factor))

            for counter in range(augmentation_count):
                insertion_count = random.randint(1, max_insertions)
                augmented_code, was_augmented = inject_dead_code(full_code, max_insertions=insertion_count)

                if was_augmented:
                    total_augmentation_count += 1
                    if saved_augmented_count < number_of_augmented_code_saving_threshold:
                        saved_augmented_count += 1
                        output_txt_path = os.path.join(augmentation_folder, f"{student_id}_{assessment_id}_augmented_{counter}.txt")
                        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                            txt_file.write(augmented_code)
                else:
                    student_embeddings[student_id][assessment_id]["augmented"].append({
                        "embedding": original_embedding.tolist(),
                        "valid": True
                    })
                    continue

                augmented_embedding = get_embedding_for_long_code(augmented_code, tokenizer, model, device)
                student_embeddings[student_id][assessment_id]["augmented"].append({
                    "embedding": augmented_embedding.tolist(),
                    "valid": was_augmented
                })

    print(f"Total augmentation count {total_augmentation_count}")

    all_assessment_ids = set()
    for scores in student_embeddings.values():
        all_assessment_ids.update(scores.keys())

    for student_id in student_embeddings:
        for assessment_id in all_assessment_ids:
            if assessment_id not in student_embeddings[student_id]:
                student_embeddings[student_id][assessment_id] = {
                    "original": {
                        "embedding": empty_code_embedding,
                        "valid": True
                    },
                    "augmented": [
                        {
                            "embedding": empty_code_embedding,
                            "valid": True
                        }
                        for _ in range(augmentation_count)
                    ],
                    "assessment_section": last_seen_assessment_section##not important 
                }

    rows = []
    for student_id, assessments in student_embeddings.items():
        for assessment_id, data in assessments.items():
            rows.append({
                "student_id": student_id,
                "assessment_id": assessment_id,
                "assessment_section": data["assessment_section"],
                "type": "original",
                "embedding": json.dumps(data["original"]["embedding"]),
                "valid": data["original"]["valid"]
            })

            for i, aug_info in enumerate(data["augmented"]):
                rows.append({
                    "student_id": student_id,
                    "assessment_id": assessment_id,
                    "assessment_section": data["assessment_section"],
                    "type": f"augmented_{i}",
                    "embedding": json.dumps(aug_info["embedding"]),
                    "valid": aug_info["valid"]
                })

    df_long = pd.DataFrame(rows)
    df_long["embedding"] = df_long["embedding"].apply(json.loads)
    df_long["column_base"] = df_long["assessment_id"] + "_" + df_long["type"]
    df_long["valid"] = df_long["valid"].astype(int)

    df_embed = df_long.pivot(index="student_id", columns="column_base", values="embedding")
    df_long["valid_flag"] = df_long["column_base"] + "_valid"
    df_status = df_long.pivot(index="student_id", columns="valid_flag", values="valid")

    df_wide = pd.concat([df_embed, df_status], axis=1)
    df_wide.columns.name = None
    df_wide.reset_index(inplace=True)

    student_to_section = df_long.groupby("student_id")["assessment_section"].first().reset_index()
    df_final = student_to_section.merge(df_wide, on="student_id")

    output_csv_path = os.path.join(dataset_folder, f"all_{assessment_type}_embeddings.csv")
    df_final.to_csv(output_csv_path, index=False)
    print(f"✅ {assessment_type.title()} embeddings saved to {output_csv_path}")
