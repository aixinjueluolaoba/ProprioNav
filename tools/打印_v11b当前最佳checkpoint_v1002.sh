#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/diana/fishing/blind_nav_rl
SUMMARY=$ROOT/runs/目标8宏动作库v11b_checkpoint评估_v1002/checkpoint_eval_summary.csv

rtk ssh v1002 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python3 - <<'PY'
import csv
from pathlib import Path

summary = Path('$SUMMARY')
rows = list(csv.DictReader(summary.open()))
stage_to_state = {
    'stage1': 'observable12_target8_macro_library_v11_stage1',
    'stage2a': 'observable12_target8_macro_library_v11_stage2a',
    'stage2': 'observable12_target8_macro_library_v11_stage2',
    'stage3': 'observable12_target8_macro_library_v11_stage3',
}

def f(row, key):
    return float(row.get(key) or 0.0)

complete = [
    row for row in rows
    if f(row, 'normal_success') >= 0.999999 and f(row, 'pressure_success') >= 0.999999
]
pool = complete or rows
target = min(
    pool,
    key=lambda row: (
        -(1 if row in complete else 0),
        -(f(row, 'pressure_success') + f(row, 'normal_success')),
        f(row, 'pressure_avg_steps') + f(row, 'normal_avg_steps'),
        f(row, 'pressure_avg_collisions'),
        int(row['checkpoint']),
    ),
)
model = Path(target['model'])
print(f'selection_policy=complete_success_then_min_total_steps_then_min_pressure_collisions_then_earliest_checkpoint')
print('stage=' + target['stage'])
print('checkpoint=' + target['checkpoint'])
print('experiment=' + model.parents[1].name)
print('state_mode=' + stage_to_state[target['stage']])
print(f'model_path={model}')
for key in [
    'normal_success',
    'normal_avg_steps',
    'normal_final_angle',
    'normal_recovery_rate',
    'pressure_success',
    'pressure_avg_steps',
    'pressure_final_angle',
    'pressure_recovery_rate',
    'pressure_avg_collisions',
]:
    print(f'{key}={target[key]}')
PY"
