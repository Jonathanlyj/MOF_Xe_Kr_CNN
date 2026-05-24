import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import cycle, combinations
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score


class RetNet(nn.Module):
    def __init__(self, extra_dim=2, channel=1):
        """
        extra_dim: number of features in your 'extras' vector (PLD, LCD, density, etc.)
        """
        super().__init__()

        # -----------------------------
        #   1) Convolutional Blocks
        # -----------------------------
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=channel, out_channels=12, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=12),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=24),
            nn.LeakyReLU(),
        )
        # Matches channels for residual
        self.adjust_channels = nn.Conv3d(
            in_channels=12, out_channels=24, kernel_size=1, 
            padding=0, padding_mode='circular', bias=False
        )
        self.max1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=32, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(),
        )
        self.max2 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=64),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=120, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=120),
            nn.LeakyReLU(),
        )

        # -----------------------------
        #   2) Flatten + FC Layers
        # -----------------------------
        self.fc1 = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(120000, 1176),
            nn.BatchNorm1d(num_features=1176),
            nn.LeakyReLU(),

            nn.Linear(1176, 84),
            nn.BatchNorm1d(num_features=84),
            nn.LeakyReLU(),
        )

        # 
        # The final fully-connected block must handle 
        # 84 + extra_dim input features
        #
        self.fc2 = nn.Sequential(
            nn.Linear(84 + extra_dim, 20),
            nn.BatchNorm1d(num_features=20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x, extras):
        """
        x      : 3D image input (batch_size, 1, D, H, W)
        extras : additional features (batch_size, extra_dim)
        """
        # -- conv1, conv2 with residual
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x1 = self.adjust_channels(x1)
        x2 += x1  # Residual connection

        # -- maxpool + conv3
        x3 = self.max1(x2)
        x3 = self.conv3(x3)

        # -- maxpool + conv4, conv5
        x4 = self.max2(x3)
        x4 = self.conv4(x4)
        x4 = self.conv5(x4)

        # -- Flatten  # or use self.fc1 which includes Flatten
        x4 = self.fc1(x4)

        # 
        # Concatenate all your extra features
        # e.g., if extras has shape (batch_size, extra_dim)
        #
        x4 = torch.cat([x4, extras], dim=1)

        # Final dense layers
        x4 = self.fc2(x4)
        return x4

class RetNetMultiOutput(nn.Module):
    def __init__(self, extra_dim=1, channel=1, num_outputs=6):
        """
        extra_dim: number of features in your 'extras' vector (PLD, LCD, density, etc.)
        num_outputs: number of pressures to predict (i.e., output vector length)
        """
        super().__init__()

        # -----------------------------
        #   1) Convolutional Blocks
        # -----------------------------
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=channel, out_channels=12, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=12),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=24),
            nn.LeakyReLU(),
        )
        self.adjust_channels = nn.Conv3d(
            in_channels=12, out_channels=24, kernel_size=1, 
            padding=0, padding_mode='circular', bias=False
        )
        self.max1 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=32, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(),
        )
        self.max2 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=64),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=120, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=120),
            nn.LeakyReLU(),
        )

        # -----------------------------
        #   2) Flatten + FC Layers
        # -----------------------------
        self.fc1 = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(120000, 1176),
            nn.BatchNorm1d(num_features=1176),
            nn.LeakyReLU(),

            nn.Linear(1176, 84),
            nn.BatchNorm1d(num_features=84),
            nn.LeakyReLU(),
        )

        # Output block modified for multi-output
        self.fc2 = nn.Sequential(
            nn.Linear(84 + extra_dim, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
            nn.Linear(20, num_outputs)  # <-- change from 1 → T
        )

    def forward(self, x, extras):
        """
        x      : (B, C, D, H, W)
        extras : (B, extra_dim)
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x1 = self.adjust_channels(x1)
        x2 += x1

        x3 = self.max1(x2)
        x3 = self.conv3(x3)
        x4 = self.max2(x3)
        x4 = self.conv4(x4)
        x4 = self.conv5(x4)

        x4 = self.fc1(x4)
        x4 = torch.cat([x4, extras], dim=1)
        x4 = self.fc2(x4)
        return x4  # shape: (B, T)

import torch.nn.functional as F

class RetNetSeq(nn.Module):
    def __init__(self, base_cnn: RetNet, pressure_dim: int, rnn_hidden_dim=64, rnn_type='LSTM', bidirection=False, num_layers=1, dropout_rate=0.3, softplus=False):
        super().__init__()
        self.cnn_encoder = base_cnn
        self.cnn_encoder.fc2 = nn.Identity()  # Remove original output layer
        self.softplus = softplus

        self.dropout_rate = dropout_rate

        # Project pressure inputs with an MLP + dropout

        self.pressure_proj = nn.Sequential(
            nn.Linear(pressure_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 32),
        )

        rnn_input_dim = 84 + 32
        rnn_output_dim = rnn_hidden_dim * 2 if bidirection else rnn_hidden_dim

        self.rnn_type = rnn_type
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=rnn_hidden_dim, batch_first=True,
                              bidirectional=bidirection, num_layers=num_layers, dropout=self.dropout_rate if num_layers > 1 else 0)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_hidden_dim, batch_first=True,
                               bidirectional=bidirection, num_layers=num_layers, dropout=self.dropout_rate if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        self.output_head = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(rnn_output_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),

        )
    def forward(self, x, extras, pressures):
        B, T, P = pressures.shape

        # CNN encoder
        x1 = self.cnn_encoder.conv1(x)
        x2 = self.cnn_encoder.conv2(x1)
        x1 = self.cnn_encoder.adjust_channels(x1)
        cnn_feat = x2 + x1
        cnn_feat = self.cnn_encoder.max1(cnn_feat)
        cnn_feat = self.cnn_encoder.conv3(cnn_feat)
        cnn_feat = self.cnn_encoder.max2(cnn_feat)
        cnn_feat = self.cnn_encoder.conv4(cnn_feat)
        cnn_feat = self.cnn_encoder.conv5(cnn_feat)
        cnn_feat = self.cnn_encoder.fc1(cnn_feat)  # (B, 84)

        # Repeat CNN features for each time step
        cnn_feat = cnn_feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, 84)

        # Project pressures + dropout
        pressures_flat = pressures.view(B * T, -1) 
        # print(B, T, P, pressures_flat.shape) gives: 64 6 2 torch.Size([384, 2])
        pressures_proj = self.pressure_proj(pressures_flat) # fugacity (vector of length 2) encoded to 32
        pressures_proj = pressures_proj.view(B, T, -1)  # (B, T, 32)

        # RNN input
        rnn_input = torch.cat([cnn_feat, pressures_proj], dim=-1)  # (B, T, 116)

        # Run RNN
        rnn_out, _ = self.rnn(rnn_input)  # (B, T, H)

        # Output head
        rnn_out_flat = rnn_out.reshape(B * T, -1)
        y_hat_flat = self.output_head(rnn_out_flat)
        y_hat = y_hat_flat.view(B, T)
        if self.softplus:
            y_hat = F.softplus(y_hat) 
        return y_hat  
# class RetNetAttn(nn.Module):
#     """
#     RNN-free version: pressure-query decoder.

#     Inputs:
#       x: energy grid
#       extras: unused here (kept for API compatibility)
#       pressures: (B, T, P)  e.g. T=6 pressure points, P=2 (fugacity vector)

#     Output:
#       y_hat: (B, T) adsorption at each pressure (single gas/property)
#     """
#     def __init__(
#         self,
#         base_cnn,
#         pressure_dim: int,
#         d_model: int = 128,
#         n_heads: int = 4,
#         n_layers: int = 2,
#         dropout_rate: float = 0.3,
#         softplus: bool = False,
#         use_pressure_self_attn: bool = True,
#     ):
#         super().__init__()
#         self.cnn_encoder = base_cnn
#         self.cnn_encoder.fc2 = nn.Identity()  # Remove original output layer
#         self.softplus = softplus
#         self.dropout_rate = dropout_rate
#         self.use_pressure_self_attn = use_pressure_self_attn

#         # 1) Pressure MLP: (P -> 32)
#         self.pressure_proj = nn.Sequential(
#             nn.Linear(pressure_dim, 32),
#             nn.ReLU(),
#             nn.Dropout(self.dropout_rate),
#             nn.Linear(32, 32),
#         )

#         # 2) Project CNN feature (84) to attention width d_model
#         self.cnn_to_model = nn.Sequential(
#             nn.Linear(84, d_model),
#             nn.LayerNorm(d_model),
#         )

#         # 3) Project pressure embedding (32) to attention width d_model
#         self.pressure_to_query = nn.Sequential(
#             nn.Linear(32, d_model),
#             nn.LayerNorm(d_model),
#         )

#         # 4) Decoder blocks: cross-attn (pressure queries attend to MOF tokens)
#         # We keep MOF tokens as length 1: (B, 1, d_model)
#         self.cross_attn_layers = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True)
#             for _ in range(n_layers)
#         ])
#         self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

#         # Optional: let the T pressure queries interact (replaces “GRU memory”)
#         if self.use_pressure_self_attn:
#             self.self_attn_layers = nn.ModuleList([
#                 nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True)
#                 for _ in range(n_layers)
#             ])
#             self.self_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

#         # Feed-forward (MLP) after attention in each layer
#         self.ff_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.LayerNorm(d_model),
#                 nn.Linear(d_model, 4 * d_model),
#                 nn.GELU(),
#                 nn.Dropout(dropout_rate),
#                 nn.Linear(4 * d_model, d_model),
#                 nn.Dropout(dropout_rate),
#             )
#             for _ in range(n_layers)
#         ])

#         # 5) Output head: produce one scalar per pressure token
#         # Use LayerNorm (not BatchNorm1d) so we do not depend on batch shape tricks.
#         self.output_head = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Dropout(self.dropout_rate),
#             nn.Linear(d_model, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 20),
#             nn.LeakyReLU(),
#             nn.Linear(20, 1),
#         )

#     def forward(self, x, extras, pressures):
#         B, T, P = pressures.shape  # e.g., (64, 6, 2)

#         # ----------------------------
#         # A) CNN encoder (unchanged)
#         # ----------------------------
#         x1 = self.cnn_encoder.conv1(x)
#         x2 = self.cnn_encoder.conv2(x1)
#         x1 = self.cnn_encoder.adjust_channels(x1)
#         cnn_feat = x2 + x1
#         cnn_feat = self.cnn_encoder.max1(cnn_feat)
#         cnn_feat = self.cnn_encoder.conv3(cnn_feat)
#         cnn_feat = self.cnn_encoder.max2(cnn_feat)
#         cnn_feat = self.cnn_encoder.conv4(cnn_feat)
#         cnn_feat = self.cnn_encoder.conv5(cnn_feat)
#         cnn_feat = self.cnn_encoder.fc1(cnn_feat)  # (B, 84)

#         # Turn MOF feature into a single "token" for attention: (B, 1, d_model)
#         mof_token = self.cnn_to_model(cnn_feat).unsqueeze(1)

#         # ----------------------------
#         # B) Build pressure queries
#         # ----------------------------
#         pressures_flat = pressures.reshape(B * T, -1)          # (B*T, P)
#         pressures_emb = self.pressure_proj(pressures_flat)     # (B*T, 32)
#         pressures_emb = pressures_emb.view(B, T, -1)           # (B, T, 32)
#         Q = self.pressure_to_query(pressures_emb)              # (B, T, d_model)

#         # ----------------------------
#         # C) Decoder: cross-attn + (optional) self-attn + FFN
#         # ----------------------------
#         for i in range(len(self.cross_attn_layers)):
#             # Cross-attention: each pressure query pulls info from the MOF token
#             # Q_new = Q + Attn(Q, mof_token)
#             Q_norm = self.cross_attn_norms[i](Q)
#             attn_out, _ = self.cross_attn_layers[i](Q_norm, mof_token, mof_token, need_weights=False)
#             Q = Q + attn_out

#             # Optional: pressures talk to each other (coupling across T=6 points)
#             if self.use_pressure_self_attn:
#                 Q_norm = self.self_attn_norms[i](Q)
#                 self_out, _ = self.self_attn_layers[i](Q_norm, Q_norm, Q_norm, need_weights=False)
#                 Q = Q + self_out

#             # Feed-forward MLP
#             Q = Q + self.ff_layers[i](Q)

#         # ----------------------------
#         # D) Output per pressure
#         # ----------------------------
#         y_hat = self.output_head(Q).squeeze(-1)  # (B, T)

#         # if self.softplus:
#         #     y_hat = F.softplus(y_hat)

#         return y_hat

class RetNetAttn(nn.Module):
    """
    RNN-free version: pressure-query decoder with latent MOF tokens.

    Inputs:
      x: energy grid
      extras: currently unused (kept for API compatibility)
      pressures: (B, T, P)  e.g. T=6 pressure points, P=2 (fugacity vector)

    Output:
      y_hat: (B, T)
    """
    def __init__(
        self,
        base_cnn,
        pressure_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.3,
        softplus: bool = False,
        use_pressure_self_attn: bool = True,
        n_mof_tokens: int = 4,   # <-- NEW: number of latent tokens
    ):
        super().__init__()
        self.cnn_encoder = base_cnn
        self.cnn_encoder.fc2 = nn.Identity()
        self.softplus = softplus
        self.dropout_rate = dropout_rate
        self.use_pressure_self_attn = use_pressure_self_attn
        self.n_mof_tokens = n_mof_tokens

        # 1) Pressure MLP: (P -> 32)
        self.pressure_proj = nn.Sequential(
            nn.Linear(pressure_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 32),
        )

        # 2) Pressure embedding (32) -> queries (d_model)
        self.pressure_to_query = nn.Sequential(
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model),
        )

        # 3) CNN feature (84) -> latent MOF tokens (N * d_model)
        #    This creates an actual sequence for K/V: (B, N, d_model)
        self.cnn_to_tokens = nn.Sequential(
            nn.LayerNorm(84),
            nn.Linear(84, n_mof_tokens * d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.mof_token_norm = nn.LayerNorm(d_model)

        # 4) Decoder blocks
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        if self.use_pressure_self_attn:
            self.self_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True
                )
                for _ in range(n_layers)
            ])
            self.self_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout_rate),
            )
            for _ in range(n_layers)
        ])

        # 5) Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x, extras, pressures):
        B, T, P = pressures.shape

        # ----------------------------
        # A) CNN encoder -> (B, 84)
        # ----------------------------
        x1 = self.cnn_encoder.conv1(x)
        x2 = self.cnn_encoder.conv2(x1)
        x1 = self.cnn_encoder.adjust_channels(x1)
        cnn_feat = x2 + x1
        cnn_feat = self.cnn_encoder.max1(cnn_feat)
        cnn_feat = self.cnn_encoder.conv3(cnn_feat)
        cnn_feat = self.cnn_encoder.max2(cnn_feat)
        cnn_feat = self.cnn_encoder.conv4(cnn_feat)
        cnn_feat = self.cnn_encoder.conv5(cnn_feat)
        cnn_feat = self.cnn_encoder.fc1(cnn_feat)  # (B, 84)

        # ----------------------------
        # A2) Build latent MOF tokens: (B, N, d_model)
        # ----------------------------
        mof_tokens = self.cnn_to_tokens(cnn_feat).view(B, self.n_mof_tokens, -1)
        mof_tokens = self.mof_token_norm(mof_tokens)

        # ----------------------------
        # B) Build pressure queries: (B, T, d_model)
        # ----------------------------
        pressures_flat = pressures.reshape(B * T, -1)      # (B*T, P)
        pressures_emb = self.pressure_proj(pressures_flat) # (B*T, 32)
        pressures_emb = pressures_emb.view(B, T, -1)       # (B, T, 32)
        Q = self.pressure_to_query(pressures_emb)          # (B, T, d_model)

        # ----------------------------
        # C) Decoder: cross-attn + (optional) self-attn + FFN
        # ----------------------------
        for i in range(len(self.cross_attn_layers)):
            Q_norm = self.cross_attn_norms[i](Q)
            attn_out, _ = self.cross_attn_layers[i](
                Q_norm, mof_tokens, mof_tokens, need_weights=False
            )
            Q = Q + attn_out

            if self.use_pressure_self_attn:
                Q_norm = self.self_attn_norms[i](Q)
                self_out, _ = self.self_attn_layers[i](
                    Q_norm, Q_norm, Q_norm, need_weights=False
                )
                Q = Q + self_out

            Q = Q + self.ff_layers[i](Q)

        # ----------------------------
        # D) Output per pressure
        # ----------------------------
        y_hat = self.output_head(Q).squeeze(-1)  # (B, T)

        # if self.softplus:
        #     y_hat = F.softplus(y_hat)

        return y_hat

class LearningMethod:
    def __init__(self, network, optimizer, criterion, logger=None):
        self.net = network 
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger

    def train(
            self, train_loader, val_loader,
            val_loss_freq=15, epochs=1, scheduler=None,
            metric=r2_score, device=None, tb_writer=None,
            verbose=True, patience=None
            ):
        
        self.scheduler = scheduler
        self.val_loss_freq = val_loss_freq
        self.train_hist = []
        self.train_metric = []
        self.val_hist = []
        self.val_metric = []
        self.writer = tb_writer
        self.train_batch_size = train_loader.batch_size
        self.val_batch_size = val_loader.batch_size
        self.epochs = epochs
        self.early_stopping_patience = patience
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

        cycled_val_loader = cycle(val_loader)

        # ...
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_epochs = 0

        for e in range(epochs):
            epoch_train_loss = 0.0
            num_train_samples = 0
            if verbose and self.logger is not None:
                self.logger.info(f'\nEpoch: {e}')

            # Training phase
            for X_train, y_train, extras_train in train_loader:
                self.net.train()
                X_train, y_train, extras_train = (
                    X_train.to(device),
                    y_train.to(device),
                    extras_train.to(device),
                )

                self.optimizer.zero_grad()
                y_train_hat = self.net(X_train, extras_train)
                train_loss = self.criterion(input=y_train_hat.ravel(), target=y_train)

                train_loss.backward()
                self.optimizer.step()
                epoch_train_loss += train_loss.item() * X_train.size(0)
                num_train_samples += X_train.size(0)

            # Validation phase
            epoch_train_loss_avg = epoch_train_loss / num_train_samples
            self.train_hist.append(epoch_train_loss_avg)
            total_val_loss = 0.0
            total_samples = 0
            self.net.eval()
            val_metrics = []

            with torch.no_grad():
                for i, (X_val, y_val, extras_val) in enumerate(val_loader):
                    X_val, y_val, extras_val = (
                        X_val.to(device),
                        y_val.to(device),
                        extras_val.to(device),
                    )
                    y_val_hat = self.net(X_val, extras_val)
                    batch_val_loss = self.criterion(input=y_val_hat.ravel(), target=y_val).item()

                    total_val_loss += batch_val_loss * X_val.size(0)
                    total_samples += X_val.size(0)

                    # Calculate metrics
                    batch_metric = metric(input=y_val_hat.ravel(), target=y_val)
                    val_metrics.append(batch_metric)

            # Compute average validation loss and metric
            epoch_val_loss = total_val_loss / total_samples
            avg_val_metric = torch.mean(torch.tensor(val_metrics)).item()

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = self.net.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            self.logger.info(
                f'End of Epoch {e}: val_loss = {epoch_val_loss:.3f}, '
                f'train_loss = {epoch_train_loss_avg:.3f}, '
                f'best_val_loss = {best_val_loss:.3f}'
                f'val_metric = {avg_val_metric:.3f}, '
            )

            if no_improve_epochs >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {e+1} epochs. "
                    f"No improvement for {self.early_stopping_patience} consecutive epochs."
                )
                self.net.load_state_dict(best_model_state)
                break

            if scheduler:
                self.scheduler.step()

        self.logger.info('\nTraining finished!')

    @torch.no_grad()
    def predict(self, X, extras):
        self.net.eval()
        y_pred = self.net(X, extras)
        return y_pred


class SeqLearningMethod(LearningMethod):
    def train(
            self, train_loader, val_loader,
            val_loss_freq=15, epochs=1, scheduler=None,
            metric=r2_score, device=None, tb_writer=None,
            verbose=True, patience=None, pressure_input=False
            ):

        self.scheduler = scheduler
        self.val_loss_freq = val_loss_freq
        self.train_hist = []
        self.train_metric = []
        self.val_hist = []
        self.val_metric = []
        self.writer = tb_writer
        self.train_batch_size = train_loader.batch_size
        self.val_batch_size = val_loader.batch_size
        self.epochs = epochs
        self.early_stopping_patience = patience
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

        best_val_loss = float('inf')
        best_model_state = None
        no_improve_epochs = 0

        for e in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_metric = 0.0
            num_train_samples = 0
            if verbose and self.logger is not None:
                self.logger.info(f'\nEpoch: {e}')

            # Training phase
            for batch in train_loader:
                if pressure_input:
                    X_train, y_train, extras_train, pressure_train = batch
                else:
                    X_train, y_train, extras_train = batch
                    pressure_train = None
                self.net.train()
                X_train, y_train, extras_train = (
                    X_train.to(device),
                    y_train.to(device),
                    extras_train.to(device)
                )
                if pressure_input:
                    pressure_train = pressure_train.to(device)
                # print(y_train[0])
                # print(y_train[1])
                self.optimizer.zero_grad()
                if pressure_input:
                    y_train_hat = self.net(X_train, extras_train, pressure_train)
                else:
                    y_train_hat = self.net(X_train, extras_train)
                train_loss = self.criterion(y_train_hat, y_train)
                train_metric = metric(input=y_train_hat, target=y_train)

                train_loss.backward()
                self.optimizer.step()
                epoch_train_loss += train_loss.item() * X_train.size(0)
                epoch_train_metric += train_metric.item() * X_train.size(0)
                num_train_samples += X_train.size(0)

            # Validation phase
            epoch_train_loss_avg = epoch_train_loss / num_train_samples
            epoch_train_metric_avg = epoch_train_metric / num_train_samples
            self.train_hist.append(epoch_train_loss_avg)
            total_val_loss = 0.0
            total_samples = 0
            self.net.eval()
            val_metrics = []

            with torch.no_grad():
                for batch in val_loader:
                    if pressure_input:
                        X_val, y_val, extras_val, pressure_val = batch
                    else:
                        X_val, y_val, extras_val = batch
                        pressure_val = None
                    X_val, y_val, extras_val= (
                        X_val.to(device),
                        y_val.to(device),
                        extras_val.to(device),
                    )
                    if pressure_input:
                        pressure_val = pressure_val.to(device)
                    
                        y_val_hat = self.net(X_val, extras_val, pressure_val)
                    else:
                        y_val_hat = self.net(X_val, extras_val)
                    batch_val_loss = self.criterion(y_val_hat, y_val).item()

                    total_val_loss += batch_val_loss * X_val.size(0)
                    total_samples += X_val.size(0)

                    batch_metric = metric(input=y_val_hat, target=y_val)
                    val_metrics.append(batch_metric)

            epoch_val_loss = total_val_loss / total_samples
            avg_val_metric = torch.mean(torch.tensor(val_metrics)).item()

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = self.net.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            self.logger.info(
                f'End of Epoch {e}: val_loss = {epoch_val_loss:.3f}, '
                f'train_loss = {epoch_train_loss_avg:.3f}, '
                f'best_val_loss = {best_val_loss:.3f}, '
                f'train_metric = {epoch_train_metric_avg:.3f}, '
                f'val_metric = {avg_val_metric:.3f}, '
            )

            if no_improve_epochs >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {e+1} epochs. "
                    f"No improvement for {self.early_stopping_patience} consecutive epochs."
                )
                self.net.load_state_dict(best_model_state)
                break

            if scheduler:
                self.scheduler.step()

        self.logger.info('\nTraining finished!')

    @torch.no_grad()
    def predict(self, X, extras, pressures=None):
        

        self.net.eval()
        if pressures is not None:
            y_pred = self.net(X, extras, pressures)
        else:
            y_pred = self.net(X, extras)
        return y_pred


class AttnLearningMethod(LearningMethod):
    def train(
            self, train_loader, val_loader,
            val_loss_freq=15, epochs=1, scheduler=None,
            metric=r2_score, device=None, tb_writer=None,
            verbose=True, patience=None, pressure_input=False
            ):

        self.scheduler = scheduler
        self.val_loss_freq = val_loss_freq
        self.train_hist = []
        self.train_metric = []
        self.val_hist = []
        self.val_metric = []
        self.writer = tb_writer
        self.train_batch_size = train_loader.batch_size
        self.val_batch_size = val_loader.batch_size
        self.epochs = epochs
        self.early_stopping_patience = patience
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

        best_val_loss = float('inf')
        best_model_state = None
        no_improve_epochs = 0

        for e in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_metric = 0.0
            num_train_samples = 0
            if verbose and self.logger is not None:
                self.logger.info(f'\nEpoch: {e}')

            # Training phase
            for batch in train_loader:
                if pressure_input:
                    X_train, y_train, extras_train, pressure_train = batch
                else:
                    X_train, y_train, extras_train = batch
                    pressure_train = None
                self.net.train()
                X_train, y_train, extras_train = (
                    X_train.to(device),
                    y_train.to(device),
                    extras_train.to(device)
                )
                if pressure_input:
                    pressure_train = pressure_train.to(device)
                # print(y_train[0])
                # print(y_train[1])
                self.optimizer.zero_grad()
                if pressure_input:
                    y_train_hat = self.net(X_train, extras_train, pressure_train)
                else:
                    y_train_hat = self.net(X_train, extras_train)
                train_loss = self.criterion(input=y_train_hat, target=y_train)
                train_metric = metric(input=y_train_hat, target=y_train)

                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_train_loss += train_loss.item() * X_train.size(0)
                epoch_train_metric += train_metric.item() * X_train.size(0)
                num_train_samples += X_train.size(0)

            # Validation phase
            epoch_train_loss_avg = epoch_train_loss / num_train_samples
            epoch_train_metric_avg = epoch_train_metric / num_train_samples
            self.train_hist.append(epoch_train_loss_avg)
            total_val_loss = 0.0
            total_samples = 0
            self.net.eval()
            val_metrics = []

            with torch.no_grad():
                for batch in val_loader:
                    if pressure_input:
                        X_val, y_val, extras_val, pressure_val = batch
                    else:
                        X_val, y_val, extras_val = batch
                        pressure_val = None
                    X_val, y_val, extras_val= (
                        X_val.to(device),
                        y_val.to(device),
                        extras_val.to(device),
                    )
                    if pressure_input:
                        pressure_val = pressure_val.to(device)
                    
                        y_val_hat = self.net(X_val, extras_val, pressure_val)
                    else:
                        y_val_hat = self.net(X_val, extras_val)
                    batch_val_loss = self.criterion(input=y_val_hat, target=y_val).item()

                    total_val_loss += batch_val_loss * X_val.size(0)
                    total_samples += X_val.size(0)

                    batch_metric = metric(input=y_val_hat, target=y_val)
                    val_metrics.append(batch_metric)

            epoch_val_loss = total_val_loss / total_samples
            avg_val_metric = torch.mean(torch.tensor(val_metrics)).item()

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = self.net.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            self.logger.info(
                f'End of Epoch {e}: val_loss = {epoch_val_loss:.3f}, '
                f'train_loss = {epoch_train_loss_avg:.3f}, '
                f'best_val_loss = {best_val_loss:.3f}, '
                f'train_metric = {epoch_train_metric_avg:.3f}, '
                f'val_metric = {avg_val_metric:.3f}, '
            )

            if no_improve_epochs >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {e+1} epochs. "
                    f"No improvement for {self.early_stopping_patience} consecutive epochs."
                )
                self.net.load_state_dict(best_model_state)
                break

            if scheduler:
                self.scheduler.step()

        self.logger.info('\nTraining finished!')

    @torch.no_grad()
    def predict(self, X, extras, pressures=None):
        

        self.net.eval()
        if pressures is not None:
            y_pred = self.net(X, extras, pressures)
        else:
            y_pred = self.net(X, extras)
        return y_pred


class CustomDataset(Dataset):
    def __init__(self, X, y, extras, transform_X=None, transform_y=None):
        """
        X      : shape (N, 1, D, H, W) for 3D input
        y      : shape (N,)
        extras : shape (N, extra_dim) 
        """
        self.transform_X = transform_X
        self.transform_y = transform_y

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.extras = extras.astype(np.float32)  # shape: (N, extra_dim)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample_x      = torch.tensor(self.X[idx], dtype=torch.float32)
        sample_y      = torch.tensor(self.y[idx], dtype=torch.float32)
        sample_extras = torch.tensor(self.extras[idx], dtype=torch.float32)

        if self.transform_X:
            sample_x = self.transform_X(sample_x)
        if self.transform_y:
            sample_y = self.transform_y(sample_y)

        return sample_x, sample_y, sample_extras



class SeqDataset(Dataset):
    def __init__(self, X, y, extras, pressures = None, transform_X=None, transform_y=None):
        """
        X        : shape (N, 1, D, H, W) or (N, C, D, H, W) for 3D input
        y        : shape (N, T) — sequence of target values
        extras   : shape (N, extra_dim)
        pressures: shape (N, T, P) — per-time-step pressure features
        """
        self.transform_X = transform_X
        self.transform_y = transform_y

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.extras = extras.astype(np.float32)
        self.pressures = None
        if pressures is not None:
            self.pressures = pressures.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample_x       = torch.tensor(self.X[idx], dtype=torch.float32)
        sample_y       = torch.tensor(self.y[idx], dtype=torch.float32)
        sample_extras  = torch.tensor(self.extras[idx], dtype=torch.float32)
        if self.pressures is not None:
            sample_pressure = torch.tensor(self.pressures[idx], dtype=torch.float32)

        if self.transform_X:
            sample_x = self.transform_X(sample_x)
        if self.transform_y:
            sample_y = self.transform_y(sample_y)
        if self.pressures is not None:
            return sample_x, sample_y, sample_extras, sample_pressure
        else:
            return sample_x, sample_y, sample_extras


@torch.no_grad()
def init_weights(m, initialization='normal', **kwargs):
    if initialization == 'normal':
        if type(m) == nn.Linear:
            m.weight = nn.init.kaiming_normal_(m.weight, **kwargs)
    elif initialization == 'uniform':
        if type(m) == nn.Linear:
            m.weight = nn.init.kaiming_uniform_(m.weight, **kwargs)


def load_data(dir_batch, path_to_csv, target_name, index_col, size=None):
    with open(f'{dir_batch}/clean.json', 'r') as fhand:
        names = json.load(fhand)['name']
    print(dir_batch, len(names))
    df = pd.read_csv(path_to_csv, index_col=index_col)

    y = df.loc[names, target_name].values
    X = np.load(f'{dir_batch}/clean.npy', mmap_mode='r')

    if size is None:
        return X, y
    else:
        return X[:size], y[:size]
    
def load_data_id(dir_batch, path_to_csv, target_name, index_col, size=None):
    with open(f'{dir_batch}/clean.json', 'r') as fhand:
        names = json.load(fhand)['name']
    print(dir_batch, len(names))
    df = pd.read_csv(path_to_csv, index_col=index_col)
    print(df.head())
    y = df.loc[names, target_name].values
    X = np.load(f'{dir_batch}/clean.npy', mmap_mode='r')
    ids = names
    if size is None:
        return X, y, ids
    else:
        return X[:size], y[:size], ids[:size]

def load_label_id(dir_batch, path_to_csv, target_name, index_col, size=None):
    with open(f'{dir_batch}/clean.json', 'r') as fhand:
        names = json.load(fhand)['name']
    print(dir_batch, len(names))
    df = pd.read_csv(path_to_csv, index_col=index_col)
    print(df.head())
    y = df.loc[names, target_name].values
    ids = names
    if size is None:
        return  y, ids
    else:
        return  y[:size], ids[:size]



class ThresholdRelativeWeightedMAELoss(nn.Module):
    """
    Weighted MAE loss with threshold-based amplification.
    Below threshold : weight = (threshold + eps) / (true + eps)  >> 1  (amplified)
    At threshold    : weight = 1.0                                      (neutral)
    Above threshold : weight = 1.0                                      (clamped, standard MAE)
    """
    def __init__(self, threshold=5.0, eps=1e-2, sharpness=0.5):
        super().__init__()
        self.threshold = threshold
        self.eps       = eps
        self.sharpness = sharpness

    def forward(self, pred, true):
        gate   = torch.sigmoid(self.sharpness * (true - self.threshold) / self.threshold)
        denom  = gate * (self.threshold + self.eps) + (1 - gate) * (true.abs() + self.eps)
        weight = (self.threshold + self.eps) / denom
        weight = torch.clamp(weight, min=1.0)

        weighted_mae = weight * (pred - true).abs()
        return weighted_mae.mean()


class AsymmetricMAELoss(nn.Module):
    """
    Pinball / asymmetric MAE with a fixed constant weight.
    
    under_weight > 1 : underprediction (pred < target) penalised more
                       → pushes model to overpredict Kr
    under_weight = 1 : standard MAE
    """
    def __init__(self, under_weight=1.5):
        super().__init__()
        self.under_weight = under_weight

    def forward(self, input, target):
        error  = input - target
        weight = torch.where(
            error < 0,
            torch.full_like(error, self.under_weight),  # underprediction
            torch.ones_like(error)                       # overprediction
        )
        return (weight * error.abs()).mean()


class AdaptiveAsymmetricMAELoss(nn.Module):
    """
    Asymmetric MAE where the weight is derived exactly from the
    selectivity formula sel = 4 * xe / kr.

    weight = (kr + δ) / (kr - δ)   for underprediction  (always > 1)
    weight = 1.0                    for overprediction

    This ensures the loss penalises underprediction proportionally
    to how much damage it causes to the downstream selectivity.
    Naturally decays toward 1.0 at large kr values where selectivity
    is less sensitive to Kr errors.
    """
    def __init__(self, eps=1e-2):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        error = input - target
        delta = error.abs()

        over_impact  = delta / (target * (target + delta) + self.eps)
        under_impact = delta / (target * (target - delta).abs() + self.eps)
        asymmetry    = under_impact / (over_impact + self.eps)  # = (kr+δ)/(kr-δ)

        weight = torch.where(
            error < 0,
            asymmetry,                  # underprediction: adaptive weight > 1
            torch.ones_like(error)      # overprediction:  standard
        )
        return (weight * delta).mean()