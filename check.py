import os

def get_files(repo_path):
    """Returns a set of relative file paths for a given directory, ignoring .git."""
    file_set = set()
    for root, dirs, files in os.walk(repo_path):
        if '.git' in dirs:
            dirs.remove('.git')  
            
        for file in files:
            
            rel_path = os.path.relpath(os.path.join(root, file), repo_path)
            file_set.add(rel_path)
    return file_set

repo1 = "Bat_Indian_Model"
repo2 = "IndianBatsModel"
repo_merged = "IndianBatsModel-main"

print(f"Scanning {repo1} and {repo2}...")
files1 = get_files(repo1)
files2 = get_files(repo2)

print(f"Scanning {repo_merged}...")
merged_files = get_files(repo_merged)

expected_files = files1.union(files2)
missing_files = expected_files - merged_files

print("\n--- RESULTS ---")
if not missing_files:
    print("Success")
else:
    print(f"❌ Found {len(missing_files)} missing files. They are NOT in the merged repository:")
    for f in sorted(missing_files):
        print(f"  - {f}")