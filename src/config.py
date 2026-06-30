"""Central configuration for the education management modeling workflow."""

from pathlib import Path

SEED = 42
N_SAMPLES = 528
TEST_SIZE = 0.20
CV_FOLDS = 5
N_SIMULATION_RUNS = 8
AUGMENTATION_NOISE = 0.05

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

MANAGEMENT_FEATURES = [
    "number_of_students",
    "number_of_managers",
    "teaching_material_utilization",
    "teaching_facility_coverage",
    "teaching_resource_investment",
    "personnel_management_input",
    "management_equipment_input",
    "reward_punishment_innovation",
    "teaching_innovation_frequency",
    "management_mode_innovation",
    "educational_management_activity",
]

REMOTE_SENSING_FEATURES = [
    "building_density",
    "green_space_ratio",
    "surface_temperature",
]

FEATURES = MANAGEMENT_FEATURES + REMOTE_SENSING_FEATURES
TARGET = "management_efficiency"
CATEGORY = "efficiency_category"
SEMESTER = "semester_group"
UNIVERSITY = "university_id"

FEATURE_DISPLAY_NAMES = {
    "number_of_students": "Students",
    "number_of_managers": "Managers",
    "teaching_material_utilization": "Material Use",
    "teaching_facility_coverage": "Facility Coverage",
    "teaching_resource_investment": "Resource Investment",
    "personnel_management_input": "Personnel Input",
    "management_equipment_input": "Equipment Input",
    "reward_punishment_innovation": "Reward Innovation",
    "teaching_innovation_frequency": "Teaching Innovation",
    "management_mode_innovation": "Mode Innovation",
    "educational_management_activity": "Management Activity",
    "building_density": "Building Density",
    "green_space_ratio": "Green Space",
    "surface_temperature": "Surface Temp",
}

BP_MODEL_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.003,
    "alpha": 0.0005,
    "batch_size": 32,
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 35,
}
