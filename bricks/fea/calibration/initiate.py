# -------------------------------- Initiation -------------------------------- #
# Define model and data paths
model_path_outer = r''
an_path_outer = r''
model_path_inner = r''
an_path_inner = r''

# Define targets for both models
outer_target = 3.5
inner_target = 3.0
targets = [outer_target, inner_target]

model_paths = [model_path_outer, model_path_inner]
tb_paths = [an_path_outer, an_path_inner]

# Instantiate WALL without manually creating scalers
WALL2 = WALL(model_paths, tb_paths, targets, bounds)
WALL2.fit_targets()
WALL2.create_run_directory()
    
# Number of initial samples and Bayesian optimization hyperparameters
n_samples = 20
n_iter = 250
batch_size = 10  # Batch size cannot be smaller than n_samples
Nrestarts = 10  # Nrestarts cannot be smaller than n_samples

# Generate initial samples
x_values = WALL2.generate_initial_samples(n_samples=n_samples)
x_unscale = WALL2.perform_scaling('targets', 'descale', x_values)
for x in x_unscale: 
    loss = WALL2.loss_function(x)
    
X_init_single = torch.tensor(x_values)
Y_init_single = torch.tensor(np.array(loss)).reshape(-1, 1)
scaled_bounds_low = WALL2.perform_scaling('targets', 'scale', np.array(bounds)[:, 0].reshape(1, -1))
scaled_bounds_high = WALL2.perform_scaling('targets', 'scale', np.array(bounds)[:, 1].reshape(1, -1))
scaled_bounds = torch.tensor(np.vstack((scaled_bounds_low, scaled_bounds_high)).T)

# # ----------------------------- Bayesian Optimization ---------------------------- #
objective_function = WALL2.loss_function
X_init_multi, Y_init_multi = SingleBOPT(Y_init_single, X_init_single, n_iter, batch_size, scaled_bounds, Nrestarts, objective_function, WALL2)