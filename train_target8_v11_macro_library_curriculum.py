from __future__ import annotations

from train_target8_v10_curriculum import main as v10_main
import train_target8_v10_curriculum as v10
from benchmark_state_dims_10k import Experiment


v10.STAGES = [
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage1",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage1",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2a",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage2a",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage3",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage3",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
]


if __name__ == "__main__":
    v10_main()
