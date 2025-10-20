import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import pandas as pd

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
                 dataloader,
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

        self.nitems = nitems
        self.latent_dims = latent_dims
        self.hidden_layer_size = hidden_layer_size
        self.dataloader = dataloader

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
        
        # Store reference to data processor (will be set externally)
        self.data_processor = None

    def set_data_processor(self, processor):
        """
        Set the data processor to be saved with checkpoints.
        Call this after creating the model: model.set_data_processor(processor)
        """
        self.data_processor = processor

    def on_save_checkpoint(self, checkpoint):
        """
        Called by PyTorch Lightning when saving a checkpoint.
        Adds data processor state to the checkpoint.
        """
        if self.data_processor is not None:
            checkpoint['data_processor'] = self.data_processor.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """
        Called by PyTorch Lightning when loading a checkpoint.
        Restores the data processor state if available.
        """
        if 'data_processor' in checkpoint:
            if self.data_processor is None:
                # Create new processor from saved state
                self.data_processor = VoteDataProcessor.from_state_dict(checkpoint['data_processor'])
            else:
                # Update existing processor
                self.data_processor.load_state_dict(checkpoint['data_processor'])

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

    def train_dataloader(self):
        return self.dataloader

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
    
class VoteDataProcessor:
    """
    Utility to load ballot-level data and reshape into the VAE input shape.

    Expected raw table columns:
      - state, county_name, cvr_id : together form a unique key for each ballot row
      - race : identifier for the contest (becomes an "item")
      - candidate : the chosen option in that race

    The processor builds:
      - a mapping from race -> item index (0..nitems-1)
      - for each item, a mapping candidate -> class index (0..Ki-1)
      - a pivoted integer matrix of shape (N_ballots, nitems) with class indices
      - a mask of shape (N_ballots, nitems) with 1 for observed, 0 for missing

    Example usage:
      p = VoteDataProcessor("ballots.csv")
      data_tensor, mask_tensor = p.get_tensors()
      dataset = p.get_torch_dataset()
    """
    def __init__(self, filepath: str = None, df: pd.DataFrame = None,
                 key_cols=('state', 'county_name', 'cvr_id'),
                 race_col='race', candidate_col='candidate'):
        if df is None and filepath is None:
            raise ValueError('Either filepath or df must be provided')

        if df is None:
            if filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath)

        self._raw = df.copy()
        self.key_cols = list(key_cols)
        self.race_col = race_col
        self.candidate_col = candidate_col

        # build race -> item index
        races = sorted(self._raw[self.race_col].unique())
        self.race_to_idx = {r: i for i, r in enumerate(races)}
        self.idx_to_race = races
        self.nitems = len(races)

        # build per-race candidate mappings
        self.candidate_maps = {}
        self.n_classes_per_item = []
        for r in races:
            vals = sorted(self._raw.loc[self._raw[self.race_col] == r, self.candidate_col].unique())
            cmap = {c: i for i, c in enumerate(vals)}
            self.candidate_maps[r] = cmap
            self.n_classes_per_item.append(len(vals))

        # map candidate -> class index per row
        def map_row(row):
            r = row[self.race_col]
            c = row[self.candidate_col]
            return self.candidate_maps[r].get(c, 0)

        self._raw['_class_idx'] = self._raw.apply(map_row, axis=1)

        # pivot into wide format: index is the ballot key, columns are races
        pivot = self._raw.set_index(self.key_cols + [self.race_col])['_class_idx'].unstack(level=self.race_col)

        # ensure columns are in the same race order
        pivot = pivot.reindex(columns=self.idx_to_race)

        # mask of observed entries
        mask = (~pivot.isna()).astype(int)

        # fill missing with 0 (safe because mask will zero-out contribution)
        pivot_filled = pivot.fillna(0).astype(int)

        # store as tensors
        self.data_tensor = torch.from_numpy(pivot_filled.values).long()
        self.mask_tensor = torch.from_numpy(mask.values).float()

        # keep the index (ballot ids)
        self.index = pivot_filled.index

    def get_tensors(self):
        """Return (data_tensor, mask_tensor)
        data_tensor: LongTensor shape [N, nitems] with class indices (0..Ki-1)
        mask_tensor: FloatTensor shape [N, nitems] with 1.0 for observed, 0.0 for missing
        """
        return self.data_tensor, self.mask_tensor

    def get_torch_dataset(self):
        """Return a torch.utils.data.TensorDataset that yields (data, mask) per row."""
        from torch.utils.data import TensorDataset
        return TensorDataset(self.data_tensor, self.mask_tensor)

    def get_n_classes_per_item(self):
        return self.n_classes_per_item
    
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
    
    def state_dict(self):
        """
        Return a dictionary containing all the state needed to reconstruct this processor.
        This allows checkpointing of the data processor along with the model.
        """
        return {
            'key_cols': self.key_cols,
            'race_col': self.race_col,
            'candidate_col': self.candidate_col,
            'race_to_idx': self.race_to_idx,
            'idx_to_race': self.idx_to_race,
            'nitems': self.nitems,
            'candidate_maps': self.candidate_maps,
            'n_classes_per_item': self.n_classes_per_item,
            'data_tensor': self.data_tensor,
            'mask_tensor': self.mask_tensor,
            'index': self.index,
        }
    
    def load_state_dict(self, state_dict):
        """
        Load processor state from a state dictionary.
        This restores all mappings, tensors, and metadata.
        """
        self.key_cols = state_dict['key_cols']
        self.race_col = state_dict['race_col']
        self.candidate_col = state_dict['candidate_col']
        self.race_to_idx = state_dict['race_to_idx']
        self.idx_to_race = state_dict['idx_to_race']
        self.nitems = state_dict['nitems']
        self.candidate_maps = state_dict['candidate_maps']
        self.n_classes_per_item = state_dict['n_classes_per_item']
        self.data_tensor = state_dict['data_tensor']
        self.mask_tensor = state_dict['mask_tensor']
        self.index = state_dict['index']
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """
        Create a new VoteDataProcessor instance from a saved state dict.
        Usage: processor = VoteDataProcessor.from_state_dict(saved_state)
        """
        # Create empty instance without calling __init__
        instance = cls.__new__(cls)
        instance.load_state_dict(state_dict)
        return instance