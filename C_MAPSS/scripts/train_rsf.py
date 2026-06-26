from utils.utils_random_survival_forest import train_model


C_MAPSS_DIR = "../../data/C_MAPSS"


if __name__ == '__main__':
    n_estimators = [100,200,300,400]
    max_depth = [5,15,30,100,250]
    min_samples_split = [10,15,20,30,50]
    min_samples_leaf = [5,10,20]
    cv_for_grid_search = 5
    dataset_root = C_MAPSS_DIR
    sub_dataset = "FD001"
    max_rul = 125
    norm_type = "z-score"
    cluster_operations = True
    norm_by_operations = True
    use_max_rul_on_test = False
    use_max_rul_on_valid = True
    percent_of_censored_data = 0.9
    percent_of_broken_data = None
    summarize_features = True

    rmse, score = train_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        cv_for_grid_search=cv_for_grid_search,
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
        summarize_features=summarize_features,
    )
