# ---------------------------------------------------------------------------- #
#                                 Optimisation                                 #
# ---------------------------------------------------------------------------- #

def SingleBOPT(Y_init_single, X_init_single, n_iter, batch_size, bounds, Nrestarts, objective_function, wall_instance):
    gp_save_dir = os.path.join(os.path.dirname(wall_instance.config['model_directory']), 'GP_models')
    os.makedirs(gp_save_dir, exist_ok=True)

    gp_model = SingleTaskGP(X_init_single, Y_init_single)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)
    gp_model.set_train_data(inputs=X_init_single, targets=Y_init_single.flatten(), strict=False)

    for iteration in range(n_iter):
        print(f'Iteration: {iteration} Best loss = {Y_init_single.min().item():.2f}\n')
        print(f'# ------------------------------------------------------------------------ #\n')

        acq_func = LogExpectedImprovement(model=gp_model, best_f=Y_init_single.min(), maximize=False)
        new_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.tensor(bounds, dtype=torch.float32).T,
            q=1,
            num_restarts=Nrestarts,
            raw_samples=batch_size,
        )
        new_x_unnorm = wall_instance.perform_scaling('targets', 'descale', new_x)
        Y_init_single = objective_function(new_x_unnorm)
        Y_init_single = torch.tensor(Y_init_single).flatten()
        X_init_single = torch.cat((X_init_single, new_x), dim=0)  # Add posterior
        gp_model.set_train_data(inputs=X_init_single, targets=Y_init_single, strict=False)
        fit_gpytorch_mll(mll)

        wall_instance.gp_model = gp_model  # Save the GP model in the WALL instance

        # Save the WALL instance after each iteration
        wall_save_path = os.path.join(wall_instance.config["save_directory"], f'wall_instance_iter_{iteration}.pkl')
        wall_instance.save(wall_save_path)

        stop = wall_instance.early_stopping()
        if stop:
            print(f'Early stopping at iteration {iteration}')
            break

    # Final save of the WALL instance
    final_save_path = os.path.join(wall_instance.config["save_directory"], 'wall_instance_final.pkl')
    wall_instance.save(final_save_path)

    return X_init_single, Y_init_single
