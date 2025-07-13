import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
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
    stride = chunk_size // 4  # e.g., 128 if chunk_size is 512
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


# ----------------------------
# Main Embedding Pipeline
# ----------------------------

def generate_all_quiz_embeddings():
    # Paths
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    quizes_folder = os.path.join(parent_folder, "Quizzes")
    dataset_folder = os.path.join(parent_folder, "Dataset")
    augmentation_folder=os.path.join(dataset_folder, "Augmentations")
    os.makedirs(dataset_folder, exist_ok=True)
    config_path = os.path.join(dataset_folder, "config.txt")
    augmentation_count=0
    model_embedding_size = 0
    number_of_augmented_code_saving_threshold=0
    # Read config value
    with open(config_path, "r") as f:
        for line in f:
            if line.startswith("augmentation_count"):
                augmentation_count = int(line.strip().split("=")[1])
            if line.startswith("model_embedding_size"):
                model_embedding_size = int(line.strip().split("=")[1])
            if line.startswith("number_of_augmented_code_saving_threshold"):
                number_of_augmented_code_saving_threshold = int(line.strip().split("=")[1])

    print("✅ augmentation_count =", augmentation_count)
    print("✅ model_embedding_size =", model_embedding_size)
    print("✅ number_of_augmented_code_saving_threshold =", number_of_augmented_code_saving_threshold)
    # Load CodeBERT
    model_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    student_embeddings = {}

    total_augmentation_count=0
    for file_name in os.listdir(quizes_folder):
        if not file_name.endswith(".json"):
            continue

        quiz_path = os.path.join(quizes_folder, file_name)
        quiz_id = os.path.splitext(file_name)[0].split(".")[0]

        print(f"📄 Processing {file_name}...")

        with open(quiz_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved_augmented_count = 0
        for student in tqdm(data["answers"], desc=f"Embedding {quiz_id}"):
            student_id = student.get("id", "unknown")
            code_sections = [v for k, v in student.items() if k.endswith(".java")]
            full_code = "\n".join(code_sections)
            if saved_augmented_count < 3:
                output_txt_path = os.path.join(augmentation_folder, f"{student_id}_{quiz_id}_original.txt")
                with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(full_code)
                print(f"📝 Saved original code to {output_txt_path}")
            # Get original embedding
            original_embedding = get_embedding_for_long_code(full_code, tokenizer, model, device)
            if student_id not in student_embeddings:
                    student_embeddings[student_id] = {}
            if quiz_id not in student_embeddings[student_id]:
                student_embeddings[student_id][quiz_id] = {
                    "original": {},
                    "augmented": [],
                }
            # Save original
            student_embeddings[student_id][quiz_id]["original"] = {
                "embedding": original_embedding.tolist(),
                "valid": True
            }
            #  Dynamically calculate how many dummy lines to inject
            code_lines = full_code.strip().split('\n')
            code_length = len(code_lines)
            scale_factor = 0.05  # inject ~5% of code length
            max_insertions = max(1, int(code_length * scale_factor))  
           
            for counter in range (augmentation_count):
                insertion_count=random.randint(1, max_insertions)
                print(f"Selected insertion count {insertion_count}")
                # Inject dead code
                augmented_code, was_augmented = inject_dead_code(full_code,max_insertions=insertion_count)
                if was_augmented:
                    print("Dead code was injected!")
                    total_augmentation_count=total_augmentation_count+1
                    # ✅ Save augmented code for some students 
                    if saved_augmented_count < number_of_augmented_code_saving_threshold:
                        saved_augmented_count += 1
                        output_txt_path = os.path.join(augmentation_folder, f"{student_id}_{quiz_id}_augmented_{counter}.txt")
                        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                            txt_file.write(augmented_code)
                        print(f"📝 Saved augmented code to {output_txt_path}")
                else : 
                    print("No dead code was injected!")
                    student_embeddings[student_id][quiz_id]["augmented"].append({
                        "embedding": [0.0] * model_embedding_size,
                        "valid": was_augmented
                    })
                    continue
                # Get embedding
                augmented_embedding  = get_embedding_for_long_code(augmented_code, tokenizer, model, device)
                # Append to list
                 # Save both embedding and whether it was truly augmented
                student_embeddings[student_id][quiz_id]["augmented"].append({
                    "embedding": augmented_embedding.tolist(),
                    "valid": was_augmented
                })
    print(f"Total augmentation count {total_augmentation_count}")
    # Fill missing quiz data with zeros
    all_quiz_ids = set()
    for quiz_scores in student_embeddings.values():
        all_quiz_ids.update(quiz_scores.keys())

    
    for student_id in student_embeddings:
        for quiz_id in all_quiz_ids:
            if quiz_id not in student_embeddings[student_id]:
                student_embeddings[student_id][quiz_id] = {
                     "original": {
                        "embedding": [0.0] * model_embedding_size,
                        "valid": False
                    },
                    "augmented": [
                        {
                            "embedding": [0.0] * model_embedding_size,
                            "valid": False
                        }
                        for _ in range(augmentation_count)
                    ]
                }
    # Flatten into rows for CSV
    rows = []
    for student_id, quizes in student_embeddings.items():
        for quiz_id, data in quizes.items():
            # Add original embedding row
            rows.append({
                "student_id": student_id,
                "quiz_id": quiz_id,
                "type": "original",
                "embedding": json.dumps(data["original"]["embedding"]),
                "valid": data["original"]["valid"]
            })

            # Add augmented embedding rows
            for i, aug_info in enumerate(data["augmented"]):
                rows.append({
                    "student_id": student_id,
                    "quiz_id": quiz_id,
                    "type": f"augmented_{i}",
                    "embedding": json.dumps(aug_info["embedding"]),
                    "valid": aug_info["valid"]
                })

    # Create long-format DataFrame
    df_long = pd.DataFrame(rows)

    # Convert JSON strings back to Python lists
    df_long["embedding"] = df_long["embedding"].apply(json.loads)

    # Create unique column base name
    df_long["column_base"] = df_long["quiz_id"] + "_" + df_long["type"]

    # Convert boolean to int: True → 1, False → 0
    df_long["valid"] = df_long["valid"].astype(int)

    # Embedding DataFrame (student_id x embeddings)
    df_embed = df_long.pivot(index="student_id", columns="column_base", values="embedding")

    # Status DataFrame (only for augmented types)
    df_status = df_long.copy()  # include both original and augmented
    df_long["valid_flag"] = df_long["column_base"] + "_valid"
    df_status = df_long.pivot(index="student_id", columns="valid_flag", values="valid")

    # Merge embedding and status columns
    df_wide = pd.concat([df_embed, df_status], axis=1)

    # Reset index and column names
    df_wide.columns.name = None
    df_wide.reset_index(inplace=True)

    # Save final output
    output_csv_path = os.path.join(dataset_folder, "all_quiz_embeddings.csv")
    df_wide.to_csv(output_csv_path, index=False)
    print(f"✅ Lab embeddings saved to {output_csv_path}")