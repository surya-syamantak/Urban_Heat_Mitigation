from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Project directories
folders = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/data_collection",
    "src/preprocessing",
    "src/hotspot_analysis",
    "src/driver_analysis",
    "src/physics_model",
    "src/scenario_simulation",
    "src/optimization",
    "dashboard",
    "outputs/maps",
    "outputs/figures",
    "outputs/tables",
    "outputs/reports",
    "tests",
]

# Create folders
for folder in folders:
    (PROJECT_ROOT / folder).mkdir(
        parents=True,
        exist_ok=True
    )

print("Project structure created successfully.")
print(f"Project location: {PROJECT_ROOT}")