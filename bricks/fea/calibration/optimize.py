import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

def SingleBOPT(Y_init_single, X_init_single, n_iter, r_samples, Nrestarts, bounds, objective_function, wall_instance):
    """
    Perform Bayesian Optimization using a single-task Gaussian Process (GP) model.
    Args:
        Y_init_single (torch.Tensor): Initial observations of the objective function.
        X_init_single (torch.Tensor): Initial input points corresponding to the observations.
        n_iter (int): Number of iterations for the optimization loop.
        r_samples (int): Number of raw samples for the acquisition function optimization.
        bounds (list): List of tuples specifying the bounds for each dimension of the input space.
        Nrestarts (int): Number of restarts for the acquisition function optimization.
        objective_function (callable): The objective function to be minimized.
        wall_instance (object): An instance of a class that provides scaling and GP model storage functionalities.
        gp_model (SingleTaskGP, optional): Pre-trained GP model. If None, a new model will be trained.
    Returns:
        tuple: A tuple containing:
            - X_init_single (torch.Tensor): Updated input points after optimization.
            - Y_init_single (torch.Tensor): Updated observations after optimization.
    """
    
    for _ in range(n_iter):
        
        print(f'Iteration: {iteration} Best loss = {Y_init_multi.min().item():.2f}\n')
        print(f'# ------------------------------------------------------------------------ #\n')

        gp_model = SingleTaskGP(X_init_single, Y_init_single)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_mll(mll)
        gp_model.set_train_data(inputs=X_init_single, targets=Y_init_single.flatten(), strict=False)
        
        LogEI = LogExpectedImprovement(gp_model,
                                       best_f= Y_init_single.min(),
                                       maximize=False)
        
        new_x, acq_value = optimize_acqf(
            acq_function=LogEI,
            bounds=torch.tensor(bounds, dtype=torch.float32).T,
            q=1,
            num_restarts=Nrestarts,
            raw_samples=r_samples,
        )
              
        new_x_unnorm = wall_instance.perform_scaling('targets', 'descale', new_x)
        Y_new = objective_function(new_x_unnorm)
                      
        X_init_single = torch.cat((X_init_single, new_x), dim=0)
        Y_init_single = torch.cat((Y_init_single.flatten(), Y_new.clone().detach()), dim=0).flatten()

        wall_instance.gp_model = gp_model  

    return X_init_single, Y_init_single
