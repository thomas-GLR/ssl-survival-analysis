from dataset import CMAPSSDataset

C_MAPSS_DIR = "data\\C_MAPSS"


if __name__ == "__main__":
    train_dataset, test_dataset, valid_dataset = CMAPSSDataset.get_data_loaders(
        dataset_root=C_MAPSS_DIR,
        sequence_len=30,
        sub_dataset='FD001',
        norm_type='z-score',
        max_rul=125,
        cluster_operations=False,
        norm_by_operations=False,
        use_max_rul_on_test=True,
        validation_rate=0.2,
        return_id=False,
        use_only_final_on_test=True,
        loader_kwargs={'batch_size': 256}
    )

    for batch_idx, (x, y) in enumerate(train_dataset):
        print("batch:", batch_idx)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print("x dtype:", x.dtype)
        print("y dtype:", y.dtype)
