from torch import nn
import torch.nn.functional as F
import torch
import polars as pl
import lightning as L

class SparseVotesDataset(torch.utils.data.Dataset):
    """Dataset for sparse vote representation - stores only non-zero entries."""
    def __init__(self, triplets, n_voters, n_items):
        self.triplets = triplets
        self.n_voters = n_voters
        self.n_items = n_items
        # build per-row slice indices for fast lookup
        self.row_map = {}
        for i, (row, item, val) in enumerate(self.triplets):
            self.row_map.setdefault(row, []).append((item, val))
    
    def __len__(self):
        return self.n_voters
    
    def __getitem__(self, idx):
        items = self.row_map.get(idx, [])
        x = torch.zeros(self.n_items, dtype=torch.long)
        m = torch.zeros(self.n_items, dtype=torch.float32)
        for item, val in items:
            x[item] = val
            m[item] = 1.0
        return x, m

class SamplingLayer(L.LightningModule):
    """
    class that samples from the approximate posterior using the reparametrisation trick
    """
    def __init__(self):
        super(SamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, sigma):
        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)
        return mu + sigma.exp() * error

class CholeskyLayer(nn.Module):
    def __init__(self, ndim, n_samples):
        super(CholeskyLayer, self).__init__()

        self.weight = nn.Parameter(torch.eye(ndim))
        self.n_samples = n_samples

    def forward(self, theta):
        L = torch.tril(self.weight, -1) + torch.eye(self.weight.shape[0]).to(self.weight)
        L = L.repeat((self.n_samples, 1,1 ))

        theta_hat =  torch.bmm(theta, L)

        return theta_hat

class ConditionalEncoder(L.LightningModule):
    """
    Encoder network that takes the mask of missing data as additional input
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(ConditionalEncoder, self).__init__()
        input_layer = nitems*2

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_layer_size)
        self.densem = nn.Linear(hidden_layer_size, latent_dims)
        self.denses = nn.Linear(hidden_layer_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """
        # code as tfidf
        #x = x * torch.log(x.shape[0] / torch.sum(x, 0))

        # concatenate the input data with the mask of missing values
        x = torch.cat([x, m], 1)
        # calculate m and mu based on encoder weights
        out = F.elu(self.dense1(x))
        out = self.bn1(out)
        mu =  self.densem(out)
        log_sigma = self.denses(out)

        return mu, log_sigma

class Decoder(L.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int, qm: torch.Tensor=None,
                 n_classes_per_item: list = None):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        :param n_classes: if all items share the same number of classes K
        :param n_classes_per_item: optional list of length nitems giving K_i per item
        """
        super().__init__()

        # Always use per-item class counts (no uniform-K path)
        assert n_classes_per_item is not None, "n_classes_per_item must be provided (list of length nitems)"
        assert len(n_classes_per_item) == nitems, "n_classes_per_item must have length nitems"

        self.nitems = nitems
        self.latent_dims = latent_dims
        self.K_list = list(n_classes_per_item)
        self.weights_list = nn.ParameterList()
        self.bias_list = nn.ParameterList()
        for Ki in self.K_list:
            # initialize weights with Xavier for better training dynamics
            w = nn.Parameter(torch.empty((latent_dims, Ki)))
            torch.nn.init.xavier_uniform_(w)
            self.weights_list.append(w)
            # biases initialized to very small non-zero values
            b = nn.Parameter(torch.randn(Ki) * 1e-2)
            self.bias_list.append(b)
        # handle qm: expect (D, nitems) mask; store as tensor for per-item use
        if qm is None:
            self.qm = None
        else:
            self.qm = torch.Tensor(qm)

    def forward(self, x: torch.Tensor):
        """
        x: tensor with shape [..., D] (D = latent_dims). Can be [S,B,D] or [B,D].
        Returns:
          - if uniform: logits tensor shaped [..., nitems, K]
          - if variable: list of length nitems with tensors [..., Ki]
        """
        outputs = []
        for i in range(self.nitems):
            W = self.weights_list[i]
            b = self.bias_list[i]
            if self.qm is not None:
                # apply per-item Q-mask on dimensions: qm shape expected (latent_dims, nitems)
                mask = self.qm[:, i].to(W)
                W_eff = W * mask.unsqueeze(1)
            else:
                W_eff = W
            logits_i = torch.matmul(x, W_eff) + b
            outputs.append(logits_i)
        return outputs

class EmbeddingEncoder(L.LightningModule):
    """
    Embedding-based encoder that accepts integer labels per item + mask
    - x: LongTensor shape [B, nitems] with values 0..(Ki-1)
    - m: FloatTensor shape [B, nitems]
    Produces mu, log_sigma of shape [B, latent_dim]
    """
    def __init__(self, n_items: int, n_classes_per_item: list, emb_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        assert len(n_classes_per_item) == n_items
        self.n_items = n_items
        self.K_list = list(n_classes_per_item)
        self.emb_dim = emb_dim
        # per-item embeddings
        self.embeddings = nn.ModuleList([nn.Embedding(Ki, emb_dim) for Ki in self.K_list])
        # optional per-item projection
        self.item_proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(n_items)])
        # pooling -> hidden -> latent params
        self.pool_fc = nn.Linear(emb_dim, hidden_dim)
        self.mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.logsigma_fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        x: LongTensor [B, nitems]
        m: FloatTensor [B, nitems]
        """
        # B = x.shape[0]
        device = x.device
        item_embs = []
        for i in range(self.n_items):
            xi = x[:, i].long().to(device)
            emb = self.embeddings[i](xi)                # [B, emb_dim]
            emb = F.elu(self.item_proj[i](emb))         # [B, emb_dim]
            mask_i = m[:, i].unsqueeze(1).to(device)
            emb = emb * mask_i
            item_embs.append(emb)
        stacked = torch.stack(item_embs, dim=1)        # [B, nitems, emb_dim]
        mask_sum = m.sum(dim=1).unsqueeze(1).clamp(min=1.0).to(device)
        pooled = stacked.sum(dim=1) / mask_sum         # [B, emb_dim]
        hidden = F.elu(self.pool_fc(pooled))           # [B, hidden_dim]
        mu = self.mu_fc(hidden)
        log_sigma = self.logsigma_fc(hidden)
        return mu, log_sigma

class CVAE(L.LightningModule):
    """
    Neural network for the conditional variational autoencoder
    """
    def __init__(self,
                #  dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 n_classes_per_item: list = None,
                 encoder_emb_dim: int = 16,
                 beta: int = 1,
                 n_samples: int = 1,
                 cholesky: bool = False):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(CVAE, self).__init__()
        self.save_hyperparameters()

        self.nitems = nitems
        self.latent_dims = latent_dims
        self.hidden_layer_size = hidden_layer_size

        # encoder: embedding-based so it accepts integer labels and a mask
        self.encoder = EmbeddingEncoder(
            n_items=self.nitems,
            n_classes_per_item=n_classes_per_item,
            emb_dim=encoder_emb_dim,
            hidden_dim=self.hidden_layer_size,
            latent_dim=self.latent_dims,
        )

        # sampler / transform
        self.sampler = SamplingLayer()
        if cholesky:
            self.transform = CholeskyLayer(latent_dims, n_samples)
        else:
            self.transform = nn.Identity()

        # decoder (expects list of per-item logits)
        self.decoder = Decoder(nitems, latent_dims, qm, n_classes_per_item=n_classes_per_item)

        # optimization params
        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.kl = 0
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, sigma = self.encoder(x, m)

        # reshape mu and log sigma in order to take multiple samples
        mu = mu.repeat(self.n_samples, 1, 1)#.permute(1, 0, 2)  # [B x S x I]
        sigma = sigma.repeat(self.n_samples, 1, 1)#.permute(1, 0, 2)  # [B x S x I]

        z = self.sampler(mu, sigma)
        reco = self.decoder(z)

        return reco, mu, sigma, z

    def training_step(self, batch, batch_idx):
        # forward pass
        data, mask = batch
        reco, mu, sigma, z  = self(data, mask)

        loss, _ = self.loss(data, reco, mask, mu, sigma, z)

        self.log('train_loss',loss, prog_bar=True)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def loss(self, input, reco, mask, mu, sigma, z):
        # calculate log likelihood for categorical outputs
        # reco: if uniform classes -> shape [S, B, nitems, K]
        # input: integer labels in [0..K-1] shape [B, nitems]
        # reco must be a list of per-item logits: each logits_i shape [S, B, Ki]
        logll_items = []
        for i, logits_i in enumerate(reco):
            # logits_i shape [S, B, Ki]
            inp = input[:, i].unsqueeze(0).repeat(logits_i.shape[0], 1)  # [S, B]
            logp = F.log_softmax(logits_i, dim=-1)
            gathered = torch.gather(logp, -1, inp.long().unsqueeze(-1)).squeeze(-1)  # [S,B]
            # apply mask for this item
            m = mask.unsqueeze(0).repeat(logits_i.shape[0], 1, 1)[:, :, i]
            logll_items.append((gathered * m).unsqueeze(-1))
        logll = torch.cat(logll_items, dim=-1).sum(dim=-1, keepdim=True)  # [S,B,1]
        
        # calculate KL divergence
        log_q_theta_x = torch.distributions.Normal(mu, sigma.exp()).log_prob(z).sum(dim = -1, keepdim = True) # log q(Theta|X)
        log_p_theta = torch.distributions.Normal(torch.zeros_like(z).to(input), scale=torch.ones(mu.shape[2]).to(input)).log_prob(z).sum(dim = -1, keepdim = True) # log p(Theta)
        kl =  log_q_theta_x - log_p_theta # kl divergence

        # combine into ELBO
        elbo = logll - kl
        # # perform importance weighting
        with torch.no_grad():
            weight = (elbo - elbo.logsumexp(dim=0)).exp()
        
        loss = (-weight * elbo).sum(0).mean()

        return loss, weight

    def fscores(self, batch, n_mc_samples=50):
        data, mask = batch

        if self.n_samples == 1:
            mu, _ = self.encoder(data, mask)
            return mu.unsqueeze(0)
        else:
            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims))
            for i in range(n_mc_samples):
                reco, mu, sigma, z = self(data, mask)

                _, weight = self.loss(data, reco, mask, mu, sigma, z)

                idxs = torch.distributions.Categorical(probs=weight.permute(1,2,0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]

                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1, idxs_expanded).squeeze().detach() # Shape [10000, 3]
                scores[i, :, :] = output

            return scores
    
class VAEDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for CVAE training
    """
    def __init__(self, filepath: str = None,
                 key_cols=('state', 'county_name', 'cvr_id'),
                 race_col='race', candidate_col='candidate', batch_size: int = 128,
                 representation: str = 'dense',  # 'dense' (pivot) or 'sparse'
                 memmap_dir: str = None,
                 data_dtype: str = 'int16',
                 mask_dtype: str = 'uint8'):
        super().__init__()
        self.batch_size = batch_size
        self.filepath = filepath
        self.key_cols = list(key_cols)
        self.race_col = race_col
        self.candidate_col = candidate_col
        self.representation = representation
        self.memmap_dir = memmap_dir
        self.data_dtype = data_dtype
        self.mask_dtype = mask_dtype

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df_lazy = pl.scan_parquet(self.filepath)

        # ------------------------------------------------------------------
        # Detect exact duplicate rows (same voter key + race + candidate).
        # Identify races containing any such duplicates and drop them.
        # ------------------------------------------------------------------
        duplicate_key_cols = self.key_cols + [self.race_col, self.candidate_col]
        dup_rows_lazy = df_lazy.filter(pl.struct(duplicate_key_cols).is_duplicated())
        dup_races_df = dup_rows_lazy.select(self.race_col).unique()
        dup_races_list = dup_races_df.collect()[self.race_col].to_list() if dup_races_df.fetch(1).height > 0 else []

        self.dropped_races = dup_races_list
        if len(dup_races_list) > 0:
            # Materialize and save duplicate rows and dropped race list for later inspection
            dup_rows = dup_rows_lazy.collect()
            dup_rows.write_csv('duplicate_entries.csv')
            pl.DataFrame({self.race_col: dup_races_list}).write_csv('dropped_races.csv')
            print(f"Dropping {len(dup_races_list)} races with duplicate rows. Saved 'duplicate_entries.csv' and 'dropped_races.csv'.")
            # Filter out problematic races entirely
            df_lazy_filtered = df_lazy.filter(~pl.col(self.race_col).is_in(dup_races_list))
        else:
            df_lazy_filtered = df_lazy
            print("No duplicate races detected; proceeding with full dataset.")

        # Get race mappings AFTER filtering
        races = sorted(df_lazy_filtered.select(self.race_col).unique().collect()[self.race_col].to_list())
        self.race_to_idx = {r: i for i, r in enumerate(races)}
        self.idx_to_race = races
        self.nitems = len(races)

        # Build per-race candidate mappings (collect grouped data efficiently)
        race_candidates = (
            df_lazy_filtered
            .group_by(self.race_col)
            .agg(pl.col(self.candidate_col).unique().sort())
            .sort(self.race_col)
            .collect()
        )

        self.candidate_maps = {}
        self.n_classes_per_item = []
        for row in race_candidates.iter_rows(named=True):
            r = row[self.race_col]
            vals = row[self.candidate_col]
            cmap = {c: i for i, c in enumerate(vals)}
            self.candidate_maps[r] = cmap
            self.n_classes_per_item.append(len(vals))

        # Map candidate -> class index
        mapping_data = []
        for race, cmap in self.candidate_maps.items():
            for candidate, idx in cmap.items():
                mapping_data.append({
                    self.race_col: race,
                    self.candidate_col: candidate,
                    '_class_idx': idx
                })
        mapping_df = pl.DataFrame(mapping_data).lazy()

        # Join lazily - query optimization happens here
        df_with_class = (
            df_lazy_filtered
            .join(mapping_df, on=[self.race_col, self.candidate_col], how='left')
            .with_columns(pl.col('_class_idx').fill_null(0))
        )

        if self.representation == 'dense':
            # Standard pivot path - fast but requires 2x memory peak
            df_collected = df_with_class.collect()
            pivot = df_collected.pivot(
                values='_class_idx',
                index=self.key_cols,
                columns=self.race_col,
                aggregate_function='first'
            )
            race_cols = [col for col in pivot.columns if col not in self.key_cols]
            ordered_race_cols = [r for r in self.idx_to_race if r in race_cols]
            pivot = pivot.select(self.key_cols + ordered_race_cols)
            race_data_cols = ordered_race_cols
            
            # Use smaller dtypes: uint8 for binary mask, configurable int for data
            mask = pivot.select([
                pl.col(col).is_not_null().cast(pl.UInt8).alias(col) 
                for col in race_data_cols
            ])
            pivot_filled = pivot.select(
                self.key_cols + [
                    pl.col(col).fill_null(0).cast(pl.Int64).alias(col)
                    for col in race_data_cols
                ]
            )
            
            # Option: write directly to memmap to avoid keeping 2 copies in RAM
            if self.memmap_dir is not None:
                import os, numpy as np
                os.makedirs(self.memmap_dir, exist_ok=True)
                n_voters, n_races = len(pivot_filled), len(race_data_cols)
                data_path = os.path.join(self.memmap_dir, 'data.memmap')
                mask_path = os.path.join(self.memmap_dir, 'mask.memmap')
                # Write numpy arrays directly to memmap files
                data_mm = np.memmap(data_path, dtype=self.data_dtype, mode='w+', shape=(n_voters, n_races))
                mask_mm = np.memmap(mask_path, dtype=self.mask_dtype, mode='w+', shape=(n_voters, n_races))
                data_mm[:] = pivot_filled.select(race_data_cols).to_numpy().astype(self.data_dtype)
                mask_mm[:] = mask.to_numpy().astype(self.mask_dtype)
                data_mm.flush()
                mask_mm.flush()
                # Load as tensors from memmap backing
                self.data_tensor = torch.from_numpy(np.array(data_mm, copy=False)).long()
                self.mask_tensor = torch.from_numpy(np.array(mask_mm, copy=False))
                self.index = pivot_filled.select(self.key_cols)
            else:
                # Pure in-memory
                data_array = pivot_filled.select(race_data_cols).to_numpy().astype(self.data_dtype)
                mask_array = mask.to_numpy().astype(self.mask_dtype)
                self.data_tensor = torch.from_numpy(data_array).long()
                self.mask_tensor = torch.from_numpy(mask_array)
                self.index = pivot_filled.select(self.key_cols)
            
            self.dataset = torch.utils.data.TensorDataset(self.data_tensor, self.mask_tensor)
        else:
            # Sparse representation: store triplets (row, race_idx, class_idx) and build dense batch on-the-fly
            import numpy as np
            keys_df = df_with_class.select(self.key_cols).unique().sort(self.key_cols).collect()
            n_voters = keys_df.height
            key_to_row = {tuple(row[k] for k in self.key_cols): i for i, row in enumerate(keys_df.iter_rows(named=True))}
            race_index_map = {r: i for i, r in enumerate(self.idx_to_race)}
            rows_df = df_with_class.select(self.key_cols + [self.race_col, '_class_idx']).collect()
            triplets = []
            for r in rows_df.iter_rows(named=True):
                voter_key = tuple(r[k] for k in self.key_cols)
                row_idx = key_to_row[voter_key]
                ridx = race_index_map[r[self.race_col]]
                triplets.append((row_idx, ridx, r['_class_idx']))
            triplets_arr = np.array(triplets, dtype=np.int32)
            self.index = keys_df.select(self.key_cols)
            self.dataset = SparseVotesDataset(triplets_arr, n_voters, self.nitems)
            # Set dummy tensors for compatibility
            self.data_tensor = None
            self.mask_tensor = None
            print(f"Sparse mode: {len(triplets)} non-zero entries for {n_voters} voters x {self.nitems} races")
            print(f"Memory savings: ~{100*(1 - len(triplets)/(n_voters*self.nitems)):.1f}% vs dense")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=8)
    
    def test_dataloader(self):
        """
        Deterministic DataLoader (no shuffling) for evaluation/exports.
        """
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    ### Additional utility methods ###
    def get_candidate_name(self, race_idx: int, class_idx: int) -> str:
        """
        Get the candidate name for a given race and class index.
        """
        race_name = self.idx_to_race[race_idx]
        # Reverse lookup in candidate_maps
        cmap = self.candidate_maps[race_name]
        for candidate, idx in cmap.items():
            if idx == class_idx:
                return candidate
        return None
    
    def get_all_candidates_for_race(self, race_idx: int) -> dict:
        """
        Get all candidates for a given race as {class_idx: candidate_name}
        """
        race_name = self.idx_to_race[race_idx]
        cmap = self.candidate_maps[race_name]
        # Return reversed map: class_idx -> candidate_name
        return {idx: candidate for candidate, idx in cmap.items()}