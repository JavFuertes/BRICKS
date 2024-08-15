# ---------------------------------------------------------------------------- #
#                                 Model object                                 #
# ---------------------------------------------------------------------------- #
class WALL:
    def __init__(self, model_path, tb_path, scalers, target, bounds):
        """
        Initializes the WALL class with the provided configuration parameters.

        Parameters:
            model_path (str): Path to the model file.
            tb_path (str): Path to the tabulated data file.
            scalers (list of StandardScaler): List of StandardScaler objects, one for each target variable.
            target (float): Target value for optimization.
            bounds (ndarray): Array of tuples specifying the bounds for each parameter (e.g., tensile limit, tensile fracture energy).

        Attributes:
            config (dict): Configuration dictionary containing model paths, scalers, and bounds.
            state (dict): Holds the state of the system, including the psi value and a DataFrame for monitoring optimization progress.
        """
        self.config = {
            "model_directory": model_path,
            "tb_directory": tb_path,
            "save_directory": os.path.join(os.path.dirname(model_path), 'monitor'),
            "target": target,
            "scalers": {
                "targets": {
                    "tensile_limit": scalers[0],
                    "tensile_fe": scalers[1]
                },
                "loss": scalers[-1]
            },
            "bounds": bounds
        }

        os.makedirs(self.config["save_directory"], exist_ok=True)

        self.state = {
            "psi": None,
            "monitor_df": pd.DataFrame(columns=["Metric", "Total Loss", "Targets", "Psi", "Time"]),
            "loss_history": []
        }

        self.gp_model = None

    def save(self, filename):
        """
        Save the WALL object to a file, including the GP model.

        Parameters:
            filename (str): Path to the file where the object should be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a WALL object from a file.

        Parameters:
            filename (str): Path to the file from which the object should be loaded.

        Returns:
            WALL: The loaded WALL object.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def fit_targets(self):
        """
        Fits the StandardScaler objects based on samples generated within the specified bounds for each material parameter.

        This method samples data within the bounds specified for each target (e.g., tensile limit, tensile fracture energy)
        and uses this data to fit the respective scalers. This ensures that the data is scaled consistently across the full
        range of possible values.
        """
        scalers = self.config["scalers"]["targets"]
        num_targets = len(scalers)
        sampled_data = np.zeros((100, num_targets))

        for i in range(num_targets):
            low, high = self.config["bounds"][i]
            sampled_data[:, i] = np.random.uniform(low, high, size=100)

        for i, scaler_key in enumerate(scalers.keys()):
            scalers[scaler_key].fit(sampled_data[:, i].reshape(-1, 1))

    def fit_losses(self, Y_init_single):
        """
        Fits the StandardScaler objects based on samples generated within the specified bounds for each material parameter.

        This method samples data within the bounds specified for each target (e.g., tensile limit, tensile fracture energy)
        and uses this data to fit the respective scalers. This ensures that the data is scaled consistently across the full
        range of possible values.
        """
        scaler = self.config["scalers"]["loss"]
        scaler.fit(Y_init_single.reshape(-1, 1))

    def perform_scaling(self, family, mode, x_values):
        x_scaled = []
        if family == 'loss':
            variables = x_values[:, 0].reshape(-1, 1)
            scaler = self.config["scalers"]["loss"]
            scaled_values = self.scaler(mode, variables, scaler)
            x_scaled.append(scaled_values)
        if family == 'targets':
            for i, scaler_type in enumerate(list(self.config["scalers"]["targets"].keys())):
                variables = x_values[:, i].reshape(-1, 1)
                scaler = self.config["scalers"]["targets"][scaler_type]
                scaled_values = self.scaler(mode, variables, scaler)
                x_scaled.append(scaled_values)
        return torch.tensor(np.array(x_scaled).T)

    def generate_initial_samples_random(self, n_samples):
        """
        Generate initial samples for Bayesian optimization based on bounds.

        Parameters:
            n_samples (int): Number of samples to generate.

        Returns:
            ndarray: Array of sampled data points within the specified bounds.

        This method generates a specified number of initial samples by uniformly sampling within the bounds
        for each parameter. These samples are used as initial data points for the Bayesian optimization process.
        """
        bounds = self.config["bounds"]
        x_list = []
        for _ in range(n_samples):
            sample = np.random.uniform(bounds[:, 0], bounds[:, 1])
            x_list.append(sample)
        x_values = np.array(x_list)
        x_scaled = self.perform_scaling('targets', 'scale', x_values)
        return x_scaled

    def generate_percentiles(self, num_percentiles):
        return np.linspace(0, 100, num_percentiles)

    def generate_initial_samples(self, n_samples):
        """
        Generate initial samples for Bayesian optimization based on bounds.

        This method generates a specified number of initial samples by uniformly sampling within the bounds
        for each parameter. These samples are used as initial data points for the Bayesian optimization process.
        """
        bounds = self.config["bounds"]
        num_params = bounds.shape[0]
        num_percentiles = int(np.ceil(n_samples ** (1 / num_params)))
        percentiles = self.generate_percentiles(num_percentiles)
        grid_points = []
        for i in range(num_params):
            param_bounds = bounds[i]
            grid_points.append(np.percentile(np.linspace(param_bounds[0], param_bounds[1], 100), percentiles))

        grid_combinations = list(itertools.product(*grid_points))
        if len(grid_combinations) > n_samples:
            np.random.shuffle(grid_combinations)
            grid_combinations = grid_combinations[:n_samples]

        x_values = np.array(grid_combinations)
        x_scaled = self.perform_scaling('targets', 'scale', x_values)

        return x_scaled

    def scaler(self, mode, x_values, scaler):
        """
        Applies scaling or inverse scaling to the provided values using the specified scaler.

        Parameters:
            mode (str): Either 'scale' to normalize the data or 'descale' to reverse the normalization.
            x_values (array): The data to be scaled or descaled.
            scaler_type (str): The key for the scaler to use ('tensile_limit' or 'tensile_fe').

        Returns:
            array: The scaled or descaled values.
        """
        if mode == 'scale':
            scaled_values = scaler.transform(x_values)
        elif mode == 'descale':
            scaled_values = scaler.inverse_transform(x_values)
        return scaled_values.flatten()

    def processPsi(self, dirTS, crackwidth_threshold=1, distance_threshold=145):
        """
        Processes the tabulated data file to compute the psi value, which represents damage.

        Parameters:
            dirTS (str): Directory path to the tabulated data file.
            crackwidth_threshold (float, optional): Minimum crack width to consider. Defaults to 0.
            distance_threshold (int, optional): Distance threshold for analyzing cracks. Defaults to 120.

        This method reads the data from the tabulated file, filters it based on the provided thresholds,
        analyzes cracks, and computes the damage parameter psi.
        """
        df = process_tb(dirTS)
        step = df['Step nr.'].max()
        df_filtered = df[(df['Step nr.'] == step) & (df['Ecw1'] >= crackwidth_threshold) & (pd.notna(df['Element']))][['Element', 'Integration Point', 'X0', 'Y0', 'Ecw1']]
        cracks = analyze_cracks(df_filtered, distance_threshold)
        psi = compute_damage_parameter(cracks)
        self.state["psi"] = psi

    def run_analysis(self, mat_param):
        """
        Runs the analysis for the given material parameters.

        Parameters:
            mat_param (list): List containing the tensile limit and tensile fracture energy.

        This method sets the necessary analysis commands and parameters, saves the project, and runs the solver
        to perform the structural analysis based on the provided material parameters.
        """
        path = self.config['model_directory']
        tensile_limit = mat_param[0]
        tensile_fracture_energy = mat_param[1]
        openProject(path)

        setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(2)/LOAD/STEPS/EXPLIC/SIZES", "0.005(200)")
        setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(1)/LOAD/STEPS/EXPLIC/SIZES", "0.033(30)")

        setParameter("MATERIAL", "Jafari EMM", "CRACKI/TENSI1/TENSTR", tensile_limit)
        setParameter("MATERIAL", "Jafari EMM", "CRACKI/GF1", tensile_fracture_energy)
        saveProject()

        runSolver([])
        return None

    def loss_function(self, x_list, delta=0.5):
        if isinstance(x_list, torch.Tensor):
            x_list = x_list.detach().cpu().numpy().flatten()

        start_time = time.time()
        self.run_analysis(x_list)
        self.processPsi(self.config["tb_directory"])

        error = self.state["psi"] - self.config["target"]
        error_small = np.abs(error) <= delta
        loss = np.where(error_small, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))
        self.state["loss_history"].append(loss)

        losses = self.state["loss_history"]
        normal_lossess = np.array(self.state["loss_history"]).reshape(-1, 1) / np.max(losses)
        self.fit_losses(normal_lossess)
        scaled_losses = self.scaler('scale', normal_lossess, self.config["scalers"]["loss"])

        # ------------------------------ Monitor losses ------------------------------ #
        elapsed_time = time.time() - start_time
        time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        optmonitor_data = [
            ["Metric", "Total Loss", "Targets", "Psi", "Time"],
            ["Value", None, x_list, self.state["psi"], time_formatted]
        ]
        new_data = pd.DataFrame([optmonitor_data[1][1:]], columns=self.state["monitor_df"].columns[1:])
        self.state["monitor_df"] = pd.concat([self.state["monitor_df"], new_data], ignore_index=True)
        self.state["monitor_df"]['Total Loss'] = pd.Series(self.state["loss_history"])

        save_path = os.path.join(self.config["save_directory"], 'monitoring_df.csv')
        self.state["monitor_df"].to_csv(save_path, index=False)
        print(self.state["monitor_df"].tail())

        return scaled_losses

    def early_stopping(self, threshold=0.01):
        last_psi = self.state["monitor_df"]['Psi'].iloc[-1]
        target = self.config["target"]
        error = np.abs(last_psi - target)
        if error / target <= threshold:
            return True

    def objective_function(self, x):
        """
        Evaluates the objective function using the loss function.

        Parameters:
            x (array): A set of design parameters.

        Returns:
            float: The value of the objective function, which is the loss.

        This method is a wrapper around the loss_function to provide an interface for optimization processes.
        """
        return self.loss_function(x)
