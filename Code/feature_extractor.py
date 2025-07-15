from quiz_feature_extractor import generate_all_quiz_embeddings
from lab_feature_extractor import generate_all_lab_embeddings
from merge_features import merge_student_features
generate_quiz_features=False
generate_lab_features=False
merge_all_features=True
def generate_features():
    if generate_quiz_features:
        print("Generating Quiz features...")
        generate_all_quiz_embeddings()
        print("✅ Quiz feature generation completed.")
    if generate_lab_features:
        print("Generating Lab features...")
        generate_all_lab_embeddings()
        print("✅ Lab feature generation completed.")
    if merge_all_features:
        print("Merging all features...")
        merge_student_features()
        print("✅ Feature merging completed.")

# Run the pipeline
if __name__ == "__main__":
    generate_features()
