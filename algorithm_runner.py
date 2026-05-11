import pandas as pd
import os
import time
import json
from datetime import datetime
import pennylane as qml
from pennylane import numpy as np
from functools import reduce
import scipy
from collections import Counter, defaultdict
from ortools.sat.python import cp_model

# Load data from the datasets folder
datasets_folder = os.path.join(os.path.dirname(__file__), "datasets")
file_path = os.path.join(datasets_folder, "QAOA_toy1_24qbit.xlsx")  # Change filename as needed
compute_evolution = True  # Set to True to run multi-depth study (slow: 4000 restarts x 5 depths)

# file_path = os.path.join(datasets_folder, "QAOA_easy_39qbit.xlsx")
# file_path = os.path.join(datasets_folder, "QAOA_easy_48qbit.xlsx")

def load_timetable_from_file(file_path):
    """Load timetable data from an Excel file in the datasets folder."""
    xls = pd.ExcelFile(file_path)
    print(f"Loading data from: {file_path}")
    print(f"Available sheets: {xls.sheet_names}")
    
    df_task = pd.read_excel(file_path, sheet_name="Task_easy", header=0)
    df_task = df_task.dropna(how='all')

    df_slotconf = pd.read_excel(file_path, sheet_name="SlotConflict_easy", header=0, index_col=0)
    df_slotconf = df_slotconf.dropna(how='all').dropna(axis=1, how='all')

    df_instructor_slot = pd.read_excel(file_path, sheet_name="InstructorSlot_easy", header=0, index_col=0)
    df_instructor_slot = df_instructor_slot.dropna(how='all').dropna(axis=1, how='all')

    df_instructor_skill = pd.read_excel(file_path, sheet_name="InstructorSkill_easy", header=0, index_col=0)
    df_instructor_skill = df_instructor_skill.dropna(how='all').dropna(axis=1, how='all')

    df_instructor_quota = pd.read_excel(file_path, sheet_name="InstructorQuota_easy", header=0, index_col=0)
    df_instructor_quota = df_instructor_quota.dropna(how='all')

    df_instructor_slot = df_instructor_slot.fillna(0).astype(int)
    df_instructor_skill = df_instructor_skill.fillna(0).astype(int)
    df_slotconf = df_slotconf.fillna(0).astype(int)
    df_instructor_quota = df_instructor_quota.fillna(0).astype(int)

    task_list = df_task[["Class", "Subject", "Slot"]].to_dict("records")
    slotconf_dict = df_slotconf.to_dict()
    instructor_slot_dict = df_instructor_slot.to_dict()
    instructor_skill_dict = df_instructor_skill.to_dict()

    instructor_quota_dict = df_instructor_quota.to_dict("index")

    return {
        "task_list": task_list,
        "slot_conflict": slotconf_dict,
        "instructor_slot": instructor_slot_dict,
        "instructor_skill": instructor_skill_dict,
        "instructor_quota": instructor_quota_dict,
    }

data = load_timetable_from_file(file_path)

# Start timing after data loading
overall_start_time = time.time()

task_list = data["task_list"]
slot_conflict = data["slot_conflict"]
instructor_slot = data["instructor_slot"]
instructor_skill = data["instructor_skill"]
instructor_quota = data["instructor_quota"]

teachers = sorted(list(instructor_skill[list(instructor_skill.keys())[0]].keys()))
sections = list(range(len(task_list)))
slots = list(slot_conflict.keys())

instructor_slot_by_teacher = {t: {slot: instructor_slot[slot][t] for slot in instructor_slot} for t in teachers}
instructor_skill_by_teacher = {t: {subj: instructor_skill[subj][t] for subj in instructor_skill} for t in teachers}

feasible_vars = []
for t_idx, teacher in enumerate(teachers):
    for s_idx, section in enumerate(task_list):
        subj = section['Subject']
        slot = section['Slot']
        if instructor_skill_by_teacher[teacher].get(subj, 0) >= 5 and instructor_slot_by_teacher[teacher].get(slot, 0) >= 5:
            feasible_vars.append((t_idx, s_idx))

var_to_idx = { (t, s): idx for idx, (t, s) in enumerate(feasible_vars) }
n_qubits = len(feasible_vars)
print(feasible_vars)
print(f"Total variables/qubits: {n_qubits}")

W_section = 3
W_conflict = 2
W_quota = 1

# ============================================================================
# QAOA SECTION
# ============================================================================
print("\n" + "="*60)
print("RUNNING QAOA OPTIMIZATION")
print("="*60)

def qubit_proj(idx):
    return (1 - qml.PauliZ(idx)) / 2

section_terms = []
for s_idx in sections:
    relevant_vars = [var_to_idx[(t, s_idx)] for t in range(len(teachers)) if (t, s_idx) in var_to_idx]
    if relevant_vars:
        expr = sum([qubit_proj(idx) for idx in relevant_vars]) - 1
        section_terms.append((W_section * expr) ** 2)

conflict_terms = []
for t_idx, teacher in enumerate(teachers):
    teacher_sections = [s for (tt, s) in feasible_vars if tt == t_idx]
    for i1, s1 in enumerate(teacher_sections):
        slot1 = task_list[s1]['Slot']
        for s2 in teacher_sections[i1+1:]:
            slot2 = task_list[s2]['Slot']
            if slot_conflict.get(slot1, {}).get(slot2, 0) == 1:
                idx1 = var_to_idx[(t_idx, s1)]
                idx2 = var_to_idx[(t_idx, s2)]
                conflict_terms.append((W_conflict*(qubit_proj(idx1) + qubit_proj(idx2))) ** 2)

quota_terms = []
for t_idx, teacher in enumerate(teachers):
    relevant_vars = [var_to_idx[(t_idx, s_idx)] for (t_idx2, s_idx) in feasible_vars if t_idx2 == t_idx]
    if not relevant_vars:
        continue

    total_expr = sum([qubit_proj(idx) for idx in relevant_vars])
    qmin = instructor_quota[teacher]["Min quota"]
    qmax = instructor_quota[teacher]["Max quota"]

    expr_min = total_expr - qmin
    expr_max = total_expr - qmax

    quota_terms.append((W_quota * expr_min) ** 2)
    quota_terms.append((W_quota * expr_max) ** 2)

all_terms = [t for t in section_terms] + [t for t in conflict_terms] + [t for t in quota_terms]
cost_h = reduce(lambda a, b: a + b, all_terms)

dev = qml.device("default.qubit", wires=n_qubits)

p = 16  # Depth

def qaoa_layer(gamma, beta):
    qml.templates.TrotterProduct(cost_h, gamma, order=1, n=1)
    for i in range(n_qubits):
        qml.RX(2 * beta, wires=i)

@qml.qnode(dev)
def circuit(params):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for gamma, beta in params:
        qaoa_layer(gamma, beta)
    return qml.expval(cost_h)

# Storage for optimization history
optimization_history = []

def logging_callback(xk):
    global iteration_count
    iteration_count += 1
    fun_val = circuit(xk.reshape((p, 2)))
    print(f"Iteration {iteration_count}: x={xk}, f(x)={fun_val}")
    optimization_history.append({
        "iteration": iteration_count,
        "x": xk.tolist(),
        "f_x": float(fun_val)
    })

np.random.seed(42)
params = 0.01 * np.random.randn(p, 2)
iteration_count = 0

qaoa_start_time = time.time()

opt_result = scipy.optimize.minimize(
    lambda x: circuit(x.reshape((p, 2))), 
    params.flatten(), 
    method="COBYLA", 
    options={'disp': True}, 
    callback=logging_callback
)

qaoa_end_time = time.time()
qaoa_elapsed_time = qaoa_end_time - qaoa_start_time

print(f"Optimal QAOA cost: {opt_result.fun}")
print(f"QAOA execution time: {qaoa_elapsed_time:.2f} seconds")

# Sample from optimized QAOA circuit
dev_sample = qml.device("default.qubit", wires=n_qubits, shots=20000)

@qml.qnode(dev_sample)
def sample_circuit(params):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for gamma, beta in params:
        qaoa_layer(gamma, beta)
    return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]

qaoa_result = sample_circuit(opt_result.x.reshape((p, 2)))
qaoa_result = np.asarray(qaoa_result)
print(f"QAOA sampled results shape: {qaoa_result.shape}")

# Convert QAOA samples to QAOA format for comparison
qaoa_samples_in_qaoa_format = [[(1 - int(i.item())) // 2 for i in row] for row in qaoa_result.transpose()]

# ============================================================================
# OR-TOOLS SECTION  
# ============================================================================
print("\n" + "="*60)
print("RUNNING OR-TOOLS OPTIMIZATION")
print("="*60)

ortools_start_time = time.time()

model = cp_model.CpModel()

x = [model.NewBoolVar(f'x_{i}') for i in range(n_qubits)]

objective_terms = []

for s_idx in sections:
    relevant_vars_indices = [var_to_idx[(t, s_idx)] for t in range(len(teachers)) if (t, s_idx) in var_to_idx]
    if relevant_vars_indices:
        sum_x_i = sum(x[idx] for idx in relevant_vars_indices)
        sum_x_i_x_j = 0
        for i_rel_idx, idx1 in enumerate(relevant_vars_indices):
             for idx2 in relevant_vars_indices[i_rel_idx + 1:]:
                 z_12 = model.NewBoolVar(f'section_{s_idx}_prod_{idx1}_{idx2}')
                 model.Add(z_12 <= x[idx1])
                 model.Add(z_12 <= x[idx2])
                 model.Add(z_12 >= x[idx1] + x[idx2] - 1)
                 sum_x_i_x_j += z_12
        objective_terms.append(W_section**2 * (2 * sum_x_i_x_j - sum_x_i + 1))

for t_idx, teacher in enumerate(teachers):
    teacher_sections_indices = [s for (tt, s) in feasible_vars if tt == t_idx]
    teacher_feasible_var_indices = [(s, var_to_idx[(t_idx, s)]) for s in teacher_sections_indices]
    for i1, (s1, idx1) in enumerate(teacher_feasible_var_indices):
        slot1 = task_list[s1]['Slot']
        for (s2, idx2) in teacher_feasible_var_indices[i1+1:]:
            slot2 = task_list[s2]['Slot']
            if slot_conflict.get(slot1, {}).get(slot2, 0) == 1:
                z_12 = model.NewBoolVar(f'conflict_{t_idx}_prod_{idx1}_{idx2}')
                model.Add(z_12 <= x[idx1])
                model.Add(z_12 <= x[idx2])
                model.Add(z_12 >= x[idx1] + x[idx2] - 1)
                objective_terms.append(W_conflict**2 * (x[idx1] + x[idx2] + 2 * z_12))

# Quota objective terms: (W_quota * (sum_i - q))^2 = W_quota^2 * ((1-2q)*sum_i + 2*sum_ij + q^2)
for t_idx, teacher in enumerate(teachers):
    relevant_vars_indices = [var_to_idx[(t_idx, s_idx)] for (t_idx2, s_idx) in feasible_vars if t_idx2 == t_idx]
    if not relevant_vars_indices:
        continue

    qmin = instructor_quota[teacher]["Min quota"]
    qmax = instructor_quota[teacher]["Max quota"]

    sum_x_i = sum(x[idx] for idx in relevant_vars_indices)
    sum_x_i_x_j = 0
    for i_rel, idx1 in enumerate(relevant_vars_indices):
        for idx2 in relevant_vars_indices[i_rel + 1:]:
            z_12 = model.NewBoolVar(f'quota_{t_idx}_prod_{idx1}_{idx2}')
            model.Add(z_12 <= x[idx1])
            model.Add(z_12 <= x[idx2])
            model.Add(z_12 >= x[idx1] + x[idx2] - 1)
            sum_x_i_x_j += z_12

    # (W_quota * (sum_i - qmin))^2
    objective_terms.append(W_quota**2 * ((1 - 2 * qmin) * sum_x_i + 2 * sum_x_i_x_j + qmin**2))
    # (W_quota * (sum_i - qmax))^2
    objective_terms.append(W_quota**2 * ((1 - 2 * qmax) * sum_x_i + 2 * sum_x_i_x_j + qmax**2))

model.Minimize(sum(objective_terms))

solver = cp_model.CpSolver()
status = solver.Solve(model)

print(f"Status: {solver.StatusName(status)}")
ortools_cost = solver.ObjectiveValue()
print(f"Optimal OR-Tools cost: {ortools_cost}")

ortools_solution = [solver.Value(var) for var in x]
print(f"OR-Tools solution: {ortools_solution}")

ortools_end_time = time.time()
ortools_elapsed_time = ortools_end_time - ortools_start_time
print(f"OR-Tools execution time: {ortools_elapsed_time:.2f} seconds\n")

# ============================================================================
# COST LANDSCAPE
# ============================================================================
print("=" * 60)
print("COMPUTING COST LANDSCAPE")
print("=" * 60)

landscape_n_points = 50
gamma_range = np.linspace(-np.pi / 2, np.pi / 2, landscape_n_points)
beta_range = np.linspace(-np.pi / 2, np.pi / 2, landscape_n_points)

cost_landscape = np.zeros((landscape_n_points, landscape_n_points))
landscape_total = landscape_n_points * landscape_n_points
landscape_done = 0
for i, gamma in enumerate(gamma_range):
    for j, beta in enumerate(beta_range):
        cost_landscape[i, j] = circuit(np.array([[gamma, beta]]))
    landscape_done += landscape_n_points
    print(f"  Landscape progress: {landscape_done}/{landscape_total}", end="\r")
print()
print("Cost landscape computed.")

# ============================================================================
# EVOLUTION ANALYSIS (multi-depth parameter study)
# ============================================================================

evolution_data = {"depths": [], "restarts": 0, "best_costs": {}, "records": []}

if compute_evolution:
    evolution_depths = [1, 2, 3, 4, 5]
    evolution_restarts = 100
    evolution_records = []
    evolution_best_costs = {}

    print("=" * 60)
    print("COMPUTING EVOLUTION (DEPTHS 1-5)")
    print("=" * 60)

    for current_p in evolution_depths:
        print(f"Optimizing for depth p={current_p} with {evolution_restarts} restarts")

        best_cost_evol = float('inf')
        best_params_evol = None

        for r in range(evolution_restarts):
            if r == 0:
                init_params = 0.01 * np.random.randn(current_p, 2)
            else:
                gamma_init = np.random.uniform(0, 2 * np.pi, current_p)
                beta_init = np.random.uniform(0, np.pi, current_p)
                init_params = np.column_stack((gamma_init, beta_init))

            opt_res = scipy.optimize.minimize(
                lambda x, cp=current_p: circuit(x.reshape((cp, 2))),
                init_params.flatten(),
                method="COBYLA"
            )

            if opt_res.fun < best_cost_evol:
                best_cost_evol = opt_res.fun
                best_params_evol = opt_res.x.reshape((current_p, 2))

        print(f"Best cost found for p={current_p}: {best_cost_evol:.6f}")
        evolution_best_costs[str(current_p)] = float(best_cost_evol)

        for layer_idx, (g, b) in enumerate(best_params_evol):
            evolution_records.append({
                "depth": current_p,
                "layer": layer_idx + 1,
                "gamma_raw": float(g),
                "beta_raw": float(b),
                "gamma_pi": float(g / np.pi),
                "beta_pi": float(b / np.pi),
            })

    evolution_data = {
        "depths": evolution_depths,
        "restarts": evolution_restarts,
        "best_costs": evolution_best_costs,
        "records": evolution_records,
    }

# ============================================================================
# SAVE STATISTICS TO JSON
# ============================================================================
print("="*60)
print("SAVING RESULTS")
print("="*60)

# Prepare data for JSON serialization
stats = {
    "problem_info": {
        "file_path": file_path,
        "n_qubits": n_qubits,
        "n_teachers": len(teachers),
        "n_sections": len(task_list),
        "feasible_vars": feasible_vars,
    },
    "weights": {
        "W_section": W_section,
        "W_conflict": W_conflict,
        "W_quota": W_quota,
    },
    "qaoa": {
        "optimal_cost": float(opt_result.fun),
        "optimal_params": opt_result.x.reshape((p, 2)).tolist(),
        "solution": [int(x) for x in opt_result.x],
        "execution_time": qaoa_elapsed_time,
        "optimization_history": optimization_history,
        "samples": qaoa_samples_in_qaoa_format,
        "cost_landscape": {
            "gamma_range": gamma_range.tolist(),
            "beta_range": beta_range.tolist(),
            "values": cost_landscape.tolist(),
        },
    },
    "ortools": {
        "optimal_cost": float(ortools_cost),
        "solution": [int(x) for x in ortools_solution],
        "execution_time": ortools_elapsed_time,
        "status": solver.StatusName(status),
    },
    "evolution": evolution_data,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
}

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# Save to JSON
json_filename = f"algorithm_results_{stats['timestamp']}.json"
json_path = os.path.join(output_dir, json_filename)

with open(json_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Results saved to: {json_path}")

# ============================================================================
# SUMMARY
# ============================================================================
overall_end_time = time.time()
total_elapsed_time = overall_end_time - overall_start_time

print("\n" + "="*60)
print("OPTIMIZATION RESULTS SUMMARY")
print("="*60)
print(f"QAOA Optimal Cost: {opt_result.fun:.6f}")
print(f"OR-Tools Optimal Cost: {ortools_cost:.6f}")
print(f"QAOA Execution Time: {qaoa_elapsed_time:.2f} seconds")
print(f"OR-Tools Execution Time: {ortools_elapsed_time:.2f} seconds")
print(f"Total Execution Time: {total_elapsed_time:.2f} seconds")
print("="*60)
print(f"\nRun 'visualizer.py' to generate visualizations from: {json_filename}")
