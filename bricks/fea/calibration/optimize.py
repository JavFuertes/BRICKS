# ---------------------------------------------------------------------------- #
#                                   Optimiser                                  #
# ---------------------------------------------------------------------------- #
def SingleBOPT(Y_init_multi, X_init_multi, n_iter, batch_size, bounds, Nrestarts, objective_function, wall_instance):
    
    # Prepare the MultiTaskGP model
    model = MultiTaskGP(
        train_X=X_init_multi,
        train_Y=Y_init_multi,
        task_feature=-1,  # Last column as task index
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    for iteration in range(n_iter):
        print(f'Iteration: {iteration} Best loss = {Y_init_multi.min().item():.2f}\n')
        print(f'# ------------------------------------------------------------------------ #\n')

        acq_func = LogExpectedImprovement(model=model, best_f=Y_init_multi.min(), maximize=False)

        new_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.tensor(bounds, dtype=torch.float32).T,
            q=1,
            num_restarts=Nrestarts,
            raw_samples=batch_size,
        )

        # Split the new_x into features and task indices
        features = new_x[:, :-1]  # All columns except the last one are features
        task_indices = new_x[:, -1].long()  # The last column is the task index

        # Perform scaling and descaling as needed
        new_x_unnorm = wall_instance.perform_scaling('targets', 'descale', features)

        # Evaluate the objective function for both tasks
        Y_task_1_new = objective_function(new_x_unnorm[task_indices == 0], model_idx=0)
        Y_task_2_new = objective_function(new_x_unnorm[task_indices == 1], model_idx=1)

        # Combine new data points
        new_x_task_1 = torch.cat([features[task_indices == 0], torch.zeros((Y_task_1_new.size(0), 1))], dim=1)
        new_x_task_2 = torch.cat([features[task_indices == 1], torch.ones((Y_task_2_new.size(0), 1))], dim=1)

        X_init_multi = torch.cat([X_init_multi, new_x_task_1, new_x_task_2], dim=0)
        Y_init_multi = torch.cat([Y_init_multi, Y_task_1_new, Y_task_2_new], dim=0)

        # Update the model with new training data
        model.set_train_data(inputs=X_init_multi, targets=Y_init_multi, strict=False)
        fit_gpytorch_mll(mll)

        wall_instance.gp_model = model  # Save the GP model in the WALL instance

        # Save the WALL instance after each iteration (overwriting the previous file)
        wall_save_path = os.path.join(wall_instance.config["save_directory"], f'wall_instance_iter_{iteration}.pkl')
        wall_instance.save(wall_save_path)

        stop = wall_instance.early_stopping()
        if stop:
            print(f'Early stopping at iteration {iteration}')
            break

    return X_init_multi, Y_init_multi

