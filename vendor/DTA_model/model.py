import torch
from torch import nn



class ResMLPBlock(nn.Module):
    def __init__(self, protein_dim, drug_dim, dropout=0.1):
        super().__init__()

        combined_dim = protein_dim + drug_dim
        self.norm = nn.BatchNorm1d(combined_dim)
        self.block = nn.Sequential(
            nn.Linear(combined_dim, 2 * combined_dim),
            nn.BatchNorm1d(2 * combined_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * combined_dim, combined_dim),
            nn.BatchNorm1d(combined_dim),
        )
        self.pro_block= nn.Sequential(
            nn.Linear(combined_dim, protein_dim),
            nn.BatchNorm1d(protein_dim),
        )
        self.drug_block= nn.Sequential(
            nn.Linear(combined_dim, drug_dim),
            nn.BatchNorm1d(drug_dim),
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.protein_norm=nn.BatchNorm1d(protein_dim)
        self.drug_norm=nn.BatchNorm1d(drug_dim)
    def forward(self, fasta_emb, drug_emb):
        cat_mean = torch.cat((self.protein_norm(fasta_emb), self.drug_norm(drug_emb)), dim=-1)
        block = self.block(self.norm(cat_mean))
        out1=self.pro_block(block)
        out2=self.drug_block(block)
        fasta=self.dropout(self.act(fasta_emb + out1))
        drug=self.dropout(self.act(drug_emb + out2))
        out = torch.cat((fasta, drug), dim=-1)
        return out


class Integration(nn.Module):
    def __init__(self, drug_model, protein_model,
                 drug_out_dim=384, protein_dim=1024,
                 hidden_sizes=[1024, 512, 256, 1], dropout=0.1):
        super().__init__()

        self.drug_model = drug_model
        self.protein_model = protein_model

        combined_dim = drug_out_dim + protein_dim

        self.res1 = ResMLPBlock(protein_dim,drug_out_dim, dropout=dropout)
        self.res2 = ResMLPBlock(protein_dim,drug_out_dim, dropout=dropout)

        layers = []
        input_dim = combined_dim * 2

        for output_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, output_dim))

            if output_dim != 1:
#                layers.append(nn.BatchNorm1d(output_dim))
                layers.append(nn.GELU())
                # layers.append(nn.Dropout(dropout))

            input_dim = output_dim

        self.regressor = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_emb, fasta_emb, fasta_max_emb):

        x = self.res1(fasta_emb, drug_emb)
        y = self.res2(fasta_max_emb, drug_emb)

        out = torch.cat((x, y), dim=-1)

        out = self.regressor(out)

        return out