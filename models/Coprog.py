import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class Coprog:
    """
    Co-training-based PROGnostics (COPROG) algorithm.

    Reference: "A co-training-based approach for prediction of remaining useful
    life utilizing both failure and suspension data." Chao Hu, Byeng D. Youn, Taejin Kim, Pingfeng Wang.

    Two models are trained on complementary views of the failure data. At each
    iteration, each model attempts to self-label the most informative suspension
    sample and passes it to the *other* model (cross-training). Training stops
    when neither model finds a beneficial sample or after `T` iterations.

    :param first_model:  A torch.nn.Module for view 1.
    :param second_model: A torch.nn.Module for view 2.
    :param w1: Weight of first model in the final ensemble prediction (default 0.5).
    :param w2: Weight of second model in the final ensemble prediction (default 0.5).
    :param lr: Learning rate used when fine-tuning with a new labelled suspension sample (default 1e-3).
    :param epochs: Number of epochs for each TrainFun call (default 20).
    :param batch_size: Mini-batch size for TrainFun (default 32).
    :param device: torch.device to run on (default: cuda if available, else cpu).
    :param shuffle_dataloader: if set to True it will shuffle the data during the training otherwise no. Default = False
    :param verbose: the print level :
        - 0 mean no print
        - 1 mean print information on the training process
        - 2 mean debugging the training process
    """

    def __init__(
            self,
            first_model: nn.Module,
            second_model: nn.Module,
            w1: float = 0.5,
            w2: float = 0.5,
            lr: float = 1e-3,
            epochs: int = 20,
            batch_size: int = 32,
            device: torch.device | None = None,
            shuffle_dataloader: bool = False,
            verbose: int = 0
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep the originals so we can deep-copy them cheaply
        self.first_model = first_model.to(self.device)
        self.second_model = second_model.to(self.device)

        self.w1 = w1
        self.w2 = w2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_dataloader = shuffle_dataloader
        self.verbose = verbose

        # Trained models (set after calling .train())
        self._h1: nn.Module | None = None
        self._h2: nn.Module | None = None

    def train(
            self,
            failure_data: torch.Tensor,
            failure_label: torch.Tensor,
            suspension_data: torch.Tensor,
            iterations: int,
            suspension_pool_size: int
    ) -> None:
        """
        Full COPROG training procedure (Algorithm 1 in the paper).

        :param failure_data:        Shape (N, *feature_dims) – labelled failure set L.
        :param failure_label:       Shape (N,) or (N, 1)     – RUL labels for L.
        :param suspension_data:     Shape (M, *feature_dims) – unlabelled suspension set U.
        :param iterations:          Maximum number of co-training rounds T.
        :param suspension_pool_size: Size u of the random sub-pool U' drawn each round.
        """
        failure_data = failure_data.to(self.device)
        failure_label = failure_label.to(self.device).float()
        suspension_data = suspension_data.to(self.device)

        # Line 1 – L1 = L2 = L  (we split L into two views)
        x1, y1 = failure_data, failure_label
        x2, y2 = failure_data, failure_label

        if self.verbose > 0:
            print("Training first and second model with labeled data...")

        # Line 2 – h1 = TrainFun(L1, 1);  h2 = TrainFun(L2, 2)
        h1 = self._train_fun(copy.deepcopy(self.first_model), x1, y1, "h1")
        h2 = self._train_fun(copy.deepcopy(self.second_model), x2, y2, "h2")

        remaining_suspension = suspension_data.clone()

        # Line 3 – Repeat for T times
        for _ in range(iterations):

            # Line 4 – Create pool U' of u suspension units
            if len(remaining_suspension) == 0:
                if self.verbose > 0:
                    print("No more remaining suspension, stopping the iterations...")
                break
            pool_size = min(suspension_pool_size, len(remaining_suspension))
            pool_indices = torch.randperm(len(remaining_suspension))[:pool_size]
            suspension_pool = remaining_suspension[pool_indices]  # U'

            pi = [None, None]  # π1, π2

            # Line 5 – for j = 1 to 2
            for j, (hj, xj, yj) in enumerate([(h1, x1, y1), (h2, x2, y2)]):

                # Line 6-9 – compute Δ for every X_u ⊂ U'
                best_delta = 0.0
                best_xu: torch.Tensor | None = None
                best_label: torch.Tensor | None = None

                if self.verbose > 0:
                    print("Iterating over the suspension pool...")

                for xu in suspension_pool:
                    xu = xu.unsqueeze(0)  # (1, *dims)

                    # Line 7 – pseudo-label from current model j
                    lu_p = self._predict(hj, xu)  # L^P_u

                    if self.verbose > 1:
                        print(f"The prediction for the current xu is lu_p with the shape : {lu_p.size()}")

                    # Line 8 – train a temporary model h'j
                    model_j = copy.deepcopy(self.first_model if j == 0 else self.second_model)
                    x_aug = torch.cat([xj, xu], dim=0)

                    if self.verbose > 1:
                        print(f"Concatenate yj with lu_p with the respective shape of {yj.size()} {lu_p.size()}")
                    y_aug = torch.cat([yj, lu_p.view(1, -1)], dim=0)
                    hj_prime = self._train_fun(model_j, x_aug, y_aug, f"h{j + 1}_prime")

                    # Line 9 – Δ_{j, X_u} = MSE(hj, L) – MSE(h'j, L)
                    delta = self._confidence_measure(xj, yj, hj, hj_prime)

                    if delta > best_delta:
                        best_delta = delta
                        best_xu = xu
                        best_label = lu_p

                # Lines 11-15 – select best candidate or set π = ∅
                if best_xu is not None and best_delta > 0:
                    # Line 12 – X*_j = argmax Δ;  L*_j = h_j(X*_j)
                    # Line 13 – π_j = {(X*_j, L*_j)};  U' = U' \ π_j
                    pi[j] = (best_xu, best_label)
                    # Remove selected sample from U'
                    mask = ~(suspension_pool == best_xu).all(dim=tuple(range(1, suspension_pool.dim())))
                    suspension_pool = suspension_pool[mask]
                else:
                    pi[j] = None  # Line 15

            # Line 17 – end for j

            # Line 18 – if π1 == ∅ && π2 == ∅  exit
            if pi[0] is None and pi[1] is None:
                if self.verbose > 0:
                    print("No beneficial samples found by either model, stopping the iterations...")
                break

            # Line 19 – L1 = L1 ∪ π2;  L2 = L2 ∪ π1   (cross-labelling)
            if pi[1] is not None:
                if self.verbose > 0:
                    print("Model h1 found a beneficial sample, adding it to the training set of h2...")
                xu2, lu2 = pi[1]
                x1 = torch.cat([x1, xu2], dim=0)
                y1 = torch.cat([y1, lu2.view(1, -1)], dim=0)
            if pi[0] is not None:
                if self.verbose > 0:
                    print("Model h2 found a beneficial sample, adding it to the training set of h1...")
                xu1, lu1 = pi[0]
                x2 = torch.cat([x2, xu1], dim=0)
                y2 = torch.cat([y2, lu1.view(1, -1)], dim=0)

            # Remove newly labelled samples from U (global pool)
            for item in [p[0] for p in pi if p is not None]:
                mask = ~(remaining_suspension == item).all(
                    dim=tuple(range(1, remaining_suspension.dim()))
                )
                remaining_suspension = remaining_suspension[mask]

            if self.verbose > 0:
                print("Training first and second model with new dataset augmented by unlabeled data...")

            # Line 20 – h1 = TrainFun(L1, 1);  h2 = TrainFun(L2, 2)
            h1 = self._train_fun(copy.deepcopy(self.first_model), x1, y1, "h1")
            h2 = self._train_fun(copy.deepcopy(self.second_model), x2, y2, "h2")

        # Save final trained models
        self._h1 = h1
        self._h2 = h2

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble prediction (line 22): L^P = w1*h1(x) + w2*h2(x).

        :param x: Shape (N, *feature_dims).

        :return Tensor of shape (N,) with predicted RUL values.
        """
        if self._h1 is None or self._h2 is None:
            raise RuntimeError("Call .train() before .predict().")

        x = x.to(self.device)
        p1 = self._predict(self._h1, x)
        p2 = self._predict(self._h2, x)
        # TODO Weight have to change depending on performance of each model
        return self.w1 * p1 + self.w2 * p2

    def _train_fun(
            self,
            model: nn.Module,
            x: torch.Tensor,
            y: torch.Tensor,
            model_name: str = ""
    ) -> nn.Module:
        """
        TrainFun: fit `model` on (x, y) with MSE loss and return it.
        Equivalent to lines 2 / 8 / 20 in the pseudo-code.
        """
        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataloader)

        if self.verbose > 0:
            print(f"Training the model {model_name} for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            avg_loss = 0.

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()

            if self.verbose > 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss : {avg_loss / len(loader)}")

        return model

    @torch.no_grad()
    def _predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Return flattened predictions (shape (N,)) without tracking gradients.
        """
        model.eval()
        return model(x).view(-1)

    def _confidence_measure(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            original_model: nn.Module,
            augmented_model: nn.Module,
    ) -> float:
        """
        Δ_{j, X_u} from line 9:
            Σ(L_i − h_j(x_i))²  −  Σ(L_i − h'_j(x_i))²

        A positive value means the augmented model is *better* on the labelled
        set, so adding X_u was beneficial.

        :param x:               Labelled failure inputs.
        :param y:               Corresponding true labels.
        :param original_model:  h_j  – model before adding the suspension sample.
        :param augmented_model: h'_j – model after  adding the suspension sample.

        :return Scalar float (positive ⟹ improvement).
        """
        y_flat = y.view(-1)

        pred_orig = self._predict(original_model, x).view(-1)
        pred_aug = self._predict(augmented_model, x).view(-1)

        mse_orig = ((y_flat - pred_orig) ** 2).sum().item()
        mse_aug = ((y_flat - pred_aug) ** 2).sum().item()

        return mse_orig - mse_aug  # > 0 means augmented model is better
