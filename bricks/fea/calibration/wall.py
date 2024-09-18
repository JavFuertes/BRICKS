# ---------------------------------------------------------------------------- #
#                                  WALL Object                                 #
# ---------------------------------------------------------------------------- #
class WALL:
    def __init__(self, model_paths, an_paths, targets, bounds):
        """
        Initializes the WALL class with multiple models and targets.
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
                "loss": StandardScaler()  # One scaler for all losses
            },
            "bounds": bounds
        }

        os.makedirs(self.config["save_directory_base"], exist_ok=True)

        self.state = {
            "psi": [[] for _ in range(num_pred)],  # List of lists for psi values for each model
            "monitor_df": pd.DataFrame(columns=["Model", "Metric", "Total Loss", "Targets", "Psi", "Time"]),
            "loss_history": []  # Store the mean loss history
        }

        self.gp_model = None  # GP model should not be pickled

    def create_run_directory(self):
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        run_directory = os.path.join(self.config["save_directory_base"], f'Run_{timestamp}')
        os.makedirs(run_directory, exist_ok=True)
        self.config["save_directory"] = run_directory

    def save(self, filename):
        """
        Save the GP model, monitor_df, loss history, and scalers in an organized dictionary.
        """
        data_to_save = {
            'gp_model': self.gp_model.state_dict() if self.gp_model else None,  # Save the state_dict of the GP model
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
    def load(filename):
        """
        Load the saved dictionary and restore the state.
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
        scalers = self.config["scalers"]["targets"]
        # Fit the scalers directly on the bounds
        for i, (scaler_key, scaler) in enumerate(scalers.items()):
            low, high = self.config["bounds"][i]
            bounds_array = np.array([low, high]).reshape(-1, 1)
            scaler.fit(bounds_array)

    def fit_losses(self, Y_init_single):
        scaler = self.config["scalers"]["loss"]
        scaler.fit(Y_init_single.reshape(-1, 1))

    def perform_scaling(self, family, mode, x_values):
        x_scaled = []
        if family == 'loss':
            scaler = self.config["scalers"]["loss"]
            scaled_values = self.scaler(mode, x_values, scaler)
            x_scaled.append(scaled_values)
        elif family == 'targets':
            for i, scaler_type in enumerate(self.config["scalers"]["targets"].keys()):
                variables = x_values[:, i].reshape(-1, 1)
                scaler = self.config["scalers"]["targets"][scaler_type]
                scaled_values = self.scaler(mode, variables, scaler)
                x_scaled.append(scaled_values)
        return torch.tensor(np.array(x_scaled).T)

    def generate_initial_samples(self, n_samples):
        bounds = np.array(self.config["bounds"])
        num_params = bounds.shape[0]
        num_samples_per_param = int(np.ceil(n_samples ** (1 / num_params)))

        grid_points = [np.linspace(bounds[i, 0], bounds[i, 1], num_samples_per_param) for i in range(num_params)]
        grid_combinations = np.array(list(itertools.product(*grid_points)))

        if grid_combinations.shape[0] > n_samples:
            np.random.shuffle(grid_combinations)
            grid_combinations = grid_combinations[:n_samples]

        x_scaled = self.perform_scaling('targets', 'scale', grid_combinations)
        return x_scaled

    def scaler(self, mode, x_values, scaler):
        if mode == 'scale':
            return scaler.transform(x_values).flatten()
        elif mode == 'descale':
            return scaler.inverse_transform(x_values).flatten()

    def run_analysis(self, mat_param, model_idx):
        path = self.config['model_directories'][model_idx]
        openProject(path)
        
        # setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(1)/LOAD/STEPS/EXPLIC/SIZES", "1")
        # setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(2)/LOAD/STEPS/EXPLIC/SIZES", "0.00277(360)")

        setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(1)/LOAD/STEPS/EXPLIC/SIZES", "1")
        setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(2)/LOAD/STEPS/EXPLIC/SIZES", "1")
        
        Ey = mat_param[0]
        Ex = mat_param[1]
        G = mat_param[2]
        tl = mat_param[3]
        tfe = mat_param[4]
        coh = mat_param[5]
        phi = mat_param[6]
        gfs = mat_param[7]

        setParameter( "MATERIAL", "Jafari EMM", "ELASTI/YOUNG", [ Ey, Ex ] )
        setParameter( "MATERIAL", "Jafari EMM", "ELASTI/SHRMOD", [ G ] )
        setParameter( "MATERIAL", "Jafari EMM", "CRACKI/TENSI1/TENSTR", tl )
        setParameter( "MATERIAL", "Jafari EMM", "CRACKI/GF1", tfe )
        setParameter( "MATERIAL", "Jafari EMM", "SHEARF/PHI", phi )
        setParameter( "MATERIAL", "Jafari EMM", "SHEARF/COHESI", coh )
        setParameter( "MATERIAL", "Jafari EMM", "SHEARF/GFS", gfs )

        saveProject()
        runSolver([])

    def examine_convergence(self, dirOUT):
        
        lines = read_file(dirOUT)
        _, ncsteps = parse_lines(lines)
        return len(ncsteps)

    def processPsi(self, dirTS, crackwidth_threshold=1, distance_threshold=145):
        df = process_tb(dirTS)
        step = df['Step nr.'].max()
        df_filtered = df[(df['Step nr.'] == step) & (df['Ecw1'] >= crackwidth_threshold) & (pd.notna(df['Element']))][['Element', 'Integration Point', 'X0', 'Y0', 'Ecw1']]
        cracks = analyze_cracks(df_filtered, distance_threshold)
        psi = compute_damage_parameter(cracks)
        return psi
    
    def loss_function(self, x_list, delta=0.5):
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
        
        # Normalize and scale the loss
        losses = np.array(self.state["loss_history"]).reshape(-1, 1)
        normal_losses = losses / np.max(losses)
        self.fit_losses(normal_losses)
        scaled_losses = self.perform_scaling('loss', 'scale', normal_losses)

        # ------------------------------ Monitor losses ------------------------------ #
        elapsed_time = time.time() - start_time
        time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        # Create a single row entry with aggregated data
        new_data = pd.DataFrame({
            "Param": [x_list],
            "Total Loss": [mean_loss],
            "Targets": [self.config["targets"]],
            "Psi": [psi_values],  # Store psi values as a list
            "Time": [time_formatted]
        })
        
        # Append the row to the monitor DataFrame
        self.state["monitor_df"] = pd.concat([self.state["monitor_df"], new_data], ignore_index=True)

        # Save the monitor DataFrame
        save_path = os.path.join(self.config["save_directory"], 'dfmonitor.csv')
        self.state["monitor_df"].to_csv(save_path, index=False)
        print(self.state["monitor_df"].tail())

        return scaled_losses

    def early_stopping(self, threshold=0.01):
        last_psi = self.state["monitor_df"]['Psi'].iloc[-1]
        targets = self.config["targets"]
        
        errors = [np.abs(last_psi[i] - targets[i]) / targets[i] for i in range(len(targets))]
        if all(error <= threshold for error in errors):
            return True
        else:
            return False

    def objective_function(self, x):
        return self.loss_function(x)
