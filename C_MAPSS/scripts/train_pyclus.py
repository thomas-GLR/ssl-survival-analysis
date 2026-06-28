from C_MAPSS.utils.utils_pyclus import train_model

C_MAPSS_DIR = "../../data/C_MAPSS"

if __name__ == "__main__":
    dataset_root = C_MAPSS_DIR
    sub_dataset = "FD001"
    max_rul = 125
    norm_type = "z-score"
    cluster_operations = False
    norm_by_operations = False
    use_max_rul_on_test = False
    use_max_rul_on_valid = True
    percent_of_censored_data = 0.6
    percent_of_broken_data = None
    random_state = 42

    cv_for_grid_search = 5

    min_leaf_size = [1, 5, 10, 20]
    n_trees = [100, 200, 300, 400]
    max_death = [5,15,30,100,250]

    train_model(
        dataset_root=dataset_root,
        sub_dataset=sub_dataset,
        max_rul=max_rul,
        norm_type=norm_type,
        cluster_operations=cluster_operations,
        norm_by_operations=norm_by_operations,
        use_max_rul_on_test=use_max_rul_on_test,
        use_max_rul_on_valid=use_max_rul_on_valid,
        percent_of_censored_data=percent_of_censored_data,
        percent_of_broken_data=percent_of_broken_data,
        seed=random_state,
        cv_for_grid_search=cv_for_grid_search,
        min_leaf_size=min_leaf_size,
        n_trees=n_trees,
        max_death=max_death,
    )