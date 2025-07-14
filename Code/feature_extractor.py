from quiz_feature_extractor import generate_quiz_features
from lab_feature_extractor import generate_lab_features
from merge_features import merge_student_features
generate_quiz_features_flag=False
generate_lab_features_flag=False
merge_all_features_flag=True
def generate_features():
    if generate_quiz_features_flag:
        print("Generating Quiz features...")
        generate_quiz_features()
        print("✅ Quiz feature generation completed.")
    if generate_lab_features_flag:
        print("Generating Lab features...")
        generate_lab_features()
        print("✅ Lab feature generation completed.")
    if merge_all_features_flag:
        print("Merging all features...")
        merge_student_features()
        print("✅ Feature merging completed.")

# Run the pipeline
if __name__ == "__main__":
    generate_features()
