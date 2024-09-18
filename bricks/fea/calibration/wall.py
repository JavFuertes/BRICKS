import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import itertools
from sklearn.preprocessing import StandardScaler
from botorch.models import SingleTaskGP

class WALL:
    """
    WALL (Weighted Analysis of Loss Landscapes) class for managing multiple models and targets.
    """

    def __init__(self, model_paths: list, an_paths: list, targets: list, bounds: list):
        """
        Initialize the WALL class with multiple models and targets.

        Args:
            model_paths (list): Paths to the model directories.
            an_paths (list): Paths to the analysis directories.
            targets (list): Target values for each model.
            bounds (list): Bounds for each parameter.
        """
        num_params = len(bounds)
        num_pred = len(targets)

        self.config = {
            "model_directories": model_paths,
            "directories": an_paths,
            "save_directory_base": os.path.join(os.path.dirname(model_paths[0]), '!Results'),
            "targets": targets,
            "scalers": {
                "targets": {f'param_{i}': StandardScaler() for i in range(num_params)},
                "loss": StandardScaler()  
            },
            "bounds": bounds
        }

        os.makedirs(self.config["save_directory_base"], exist_ok=True)

        self.state = {
            "psi": [[] for _ in range(num_pred)],  
            "monitor_df": pd.DataFrame(columns=["Param", "Total Loss", "Targets", "Psi", "Time"]),
            "loss_history": []  
        }

        self.gp_model = None

    def create_run_directory(self):
        """Create a new directory for the current run."""
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        run_directory = os.path.join(self.config["save_directory_base"], f'Run_{timestamp}')
        os.makedirs(run_directory, exist_ok=True)
        self.config["save_directory"] = run_directory

    def save(self, filename: str):
        """
        Save the current state of the WALL object.

        Args:
            filename (str): Path to save the state.
        """
        data_to_save = {
            'gp_model': self.gp_model.state_dict() if self.gp_model else None,
            'monitor_df': self.state["monitor_df"],
            'loss_history': self.state["loss_history"],
            'scalers': {
                'targets': {key: scaler for key, scaler in self.config["scalers"]["targets"].items()},
                'loss': self.config["scalers"]["loss"]
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)

    @staticmethod
    def load(filename: str) -> 'WALL':
        """
        Load a saved WALL object.

        Args:
            filename (str): Path to the saved state.

        Returns:
            WALL: A new WALL instance with the loaded state.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        wall_instance = WALL([], [], [], [])
        wall_instance.gp_model = SingleTaskGP()
        wall_instance.gp_model.load_state_dict(data['gp_model'])
        wall_instance.state["monitor_df"] = data['monitor_df']
        wall_instance.state["loss_history"] = data['loss_history']
        wall_instance.config["scalers"]["targets"] = data['scalers']['targets']
        wall_instance.config["scalers"]["loss"] = data['scalers']['loss']

        return wall_instance
    
    def fit_targets(self):
        """Fit the target scalers to the bounds."""
        for i, (scaler_key, scaler) in enumerate(self.config["scalers"]["targets"].items()):
            low, high = self.config["bounds"][i]
            bounds_array = np.array([low, high]).reshape(-1, 1)
            scaler.fit(bounds_array)

    def fit_losses(self, Y_init_single: np.ndarray):
        """
        Fit the loss scaler to the initial losses.

        Args:
            Y_init_single (np.ndarray): Initial loss values.
        """
        self.config["scalers"]["loss"].fit(Y_init_single.reshape(-1, 1))

    def perform_scaling(self, family: str, mode: str, x_values: np.ndarray) -> torch.Tensor:
        """
        Scale or descale values using the appropriate scaler.

        Args:
            family (str): 'loss' or 'targets'.
            mode (str): 'scale' or 'descale'.
            x_values (np.ndarray): Values to scale or descale.

        Returns:
            torch.Tensor: Scaled or descaled values.
        """
        if family == 'loss':
            scaler = self.config["scalers"]["loss"]
            scaled_values = self.scaler(mode, x_values, scaler)
            return torch.tensor(scaled_values.reshape(-1, 1))
        elif family == 'targets':
            x_scaled = []
            for i, scaler_type in enumerate(self.config["scalers"]["targets"].keys()):
                variables = x_values[:, i].reshape(-1, 1)
                scaler = self.config["scalers"]["targets"][scaler_type]
                scaled_values = self.scaler(mode, variables, scaler)
                x_scaled.append(scaled_values)
            return torch.tensor(np.array(x_scaled).T)

    def generate_initial_samples(self, n_samples: int) -> torch.Tensor:
        """
        Generate initial samples using a grid-based approach.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Scaled initial samples.
        """
        bounds = np.array(self.config["bounds"])
        num_params = bounds.shape[0]
        num_samples_per_param = int(np.ceil(n_samples ** (1 / num_params)))

        grid_points = [np.linspace(bounds[i, 0], bounds[i, 1], num_samples_per_param) for i in range(num_params)]
        grid_combinations = np.array(list(itertools.product(*grid_points)))

        if grid_combinations.shape[0] > n_samples:
            np.random.shuffle(grid_combinations)
            grid_combinations = grid_combinations[:n_samples]

        return self.perform_scaling('targets', 'scale', grid_combinations)

    @staticmethod
    def scaler(mode: str, x_values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        """
        Scale or descale values using a given scaler.

        Args:
            mode (str): 'scale' or 'descale'.
            x_values (np.ndarray): Values to scale or descale.
            scaler (StandardScaler): Scaler to use.

        Returns:
            np.ndarray: Scaled or descaled values.
        """
        if mode == 'scale':
            return scaler.transform(x_values).flatten()
        elif mode == 'descale':
            return scaler.inverse_transform(x_values).flatten()

    def run_analysis(self, mat_param: np.ndarray, model_idx: int):
        """
        Run the analysis for a given set of material parameters.

        Args:
            mat_param (np.ndarray): Material parameters.
            model_idx (int): Index of the model to run.
        """
        path = self.config['model_directories'][model_idx]
        openProject(path)
        
        setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(1)/LOAD/STEPS/EXPLIC/SIZES", "1")
        setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(2)/LOAD/STEPS/EXPLIC/SIZES", "1")
        
        Ey, Ex, G, tl, tfe, coh, phi, gfs = mat_param

        setParameter("MATERIAL", "Jafari EMM", "ELASTI/YOUNG", [Ey, Ex])
        setParameter("MATERIAL", "Jafari EMM", "ELASTI/SHRMOD", [G])
        setParameter("MATERIAL", "Jafari EMM", "CRACKI/TENSI1/TENSTR", tl)
        setParameter("MATERIAL", "Jafari EMM", "CRACKI/GF1", tfe)
        setParameter("MATERIAL", "Jafari EMM", "SHEARF/PHI", phi)
        setParameter("MATERIAL", "Jafari EMM", "SHEARF/COHESI", coh)
        setParameter("MATERIAL", "Jafari EMM", "SHEARF/GFS", gfs)

        saveProject()
        runSolver([])

    def examine_convergence(self, dirOUT: str) -> int:
        """
        Examine the convergence of the analysis.

        Args:
            dirOUT (str): Path to the output file.

        Returns:
            int: Number of non-converged steps.
        """
        lines = read_file(dirOUT)
        _, ncsteps = parse_lines(lines)
        return len(ncsteps)

    def processPsi(self, dirTS: str, crackwidth_threshold: float = 1, distance_threshold: float = 145) -> float:
        """
        Process the analysis results to compute Psi.

        Args:
            dirTS (str): Path to the tabulated results file.
            crackwidth_threshold (float): Threshold for crack width.
            distance_threshold (float): Threshold for distance between cracks.

        Returns:
            float: Computed Psi value.
        """
        df = process_tb(dirTS)
        step = df['Step nr.'].max()
        df_filtered = df[(df['Step nr.'] == step) & (df['Ecw1'] >= crackwidth_threshold) & (pd.notna(df['Element']))][['Element', 'Integration Point', 'X0', 'Y0', 'Ecw1']]
        cracks = analyze_cracks(df_filtered, distance_threshold)
        return compute_damage_parameter(cracks)
    
    def loss_function(self, x_list: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
        """
        Compute the loss for a given set of parameters.

        Args:
            x_list (torch.Tensor): Parameters to evaluate.
            delta (float): Delta value for the loss function.

        Returns:
            torch.Tensor: Scaled loss value.
        """
        if isinstance(x_list, torch.Tensor):
            x_list = x_list.detach().cpu().numpy().flatten()

        start_time = time.time()
        
        psi_values = []
        loss_values = []
        
        for model_idx, model_dir in enumerate(self.config["model_directories"]):
            self.run_analysis(x_list, model_idx)
            dir = self.config["directories"][model_idx]
            
            try:
                nnc = self.examine_convergence(dir + '.out')
            except Exception as e:
                logging.warning(f"Convergence examination failed for model {model_idx}: {e}")
                nnc = 10

            psi = self.processPsi(dir + '.tb')
            self.state["psi"][model_idx].append(psi)
            psi_values.append(psi)

            error = psi - self.config["targets"][model_idx]
            error_small = np.abs(error) <= delta
            loss = np.where(error_small, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))
            
            if nnc > 2:
                loss *= 10 
            
            loss_values.append(loss)
        
        mean_loss = np.mean(loss_values)
        self.state["loss_history"].append(mean_loss)
        
        losses = np.array(self.state["loss_history"]).reshape(-1, 1)
        normal_losses = losses / np.max(losses)
        self.fit_losses(normal_losses)
        scaled_losses = self.perform_scaling('loss', 'scale', normal_losses)

        elapsed_time = time.time() - start_time
        time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        new_data = pd.DataFrame({
            "Param": [x_list],
            "Total Loss": [mean_loss],
            "Targets": [self.config["targets"]],
            "Psi": [psi_values],
            "Time": [time_formatted]
        })
        
        self.state["monitor_df"] = pd.concat([self.state["monitor_df"], new_data], ignore_index=True)

        save_path = os.path.join(self.config["save_directory"], 'dfmonitor.csv')
        self.state["monitor_df"].to_csv(save_path, index=False)
        print(self.state["monitor_df"].tail())

        return scaled_losses

    def early_stopping(self, threshold: float = 0.01) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            threshold (float): Threshold for relative error.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        last_psi = self.state["monitor_df"]['Psi'].iloc[-1]
        targets = self.config["targets"]
        
        errors = [np.abs(last_psi[i] - targets[i]) / targets[i] for i in range(len(targets))]
        return all(error <= threshold for error in errors)

    def objective_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Objective function for optimization.

        Args:
            x (torch.Tensor): Parameters to evaluate.

        Returns:
            torch.Tensor: Loss value.
        """
        return self.loss_function(x)