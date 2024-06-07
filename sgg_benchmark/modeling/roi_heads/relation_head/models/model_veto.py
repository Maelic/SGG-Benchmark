import torch
import torch.nn as nn

class VETOTransformer(nn.Module):
    def __init__(self, config=None, in_channels=256):
        super(VETOTransformer, self).__init__()

        self.transformer = Transformer(config=config, in_channels=in_channels)

        self.to_cls_token = nn.Identity()


    def forward(self, d,v, l, c, ctxt=None):
        x = self.transformer(d, v, l, c, ctxt)
        x = x.to(d.dtype)
        for i, dv in enumerate(self.transformer.layers):
            dv_attn, dv_ff = dv[0], dv[1]
            x = dv_attn(x) + x
            x = dv_ff(x) + x

        x_int = x[:, 0]
        xy_cls_token = self.to_cls_token(x_int)

        return xy_cls_token

class Transformer(nn.Module):
    def __init__(self, config=None, in_channels=256):
        super(Transformer, self).__init__()

        patch_size = config.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.PATCH_SIZE
        t_input_dim = config.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.T_INPUT_DIM
        enc_layers = config.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.ENC_LAYERS
        nheads = config.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.NHEADS
        emb_dropout = config.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.EMB_DROPOUT
        t_dropout = config.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.T_DROPOUT

        mlp_dim = t_input_dim * 2
        self.patch_embed = PatchEmbed(patch_size=patch_size,
                                      in_channels=in_channels, embed_dim=t_input_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, t_input_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, t_input_dim))
        self.pos_drop = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([])
        for _ in range(enc_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(t_input_dim,
                        Attention(t_input_dim=t_input_dim, nheads=nheads, dropout=t_dropout)),
                PreNorm(t_input_dim, FeedForward(dim=t_input_dim, hidden_dim=mlp_dim, dropout=0))
            ]))
    def forward(self, d, v, l, c, ctxt=None):
        x = self.patch_embed(d, v)
        l = l.unsqueeze(1)
        c = c.unsqueeze(1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        if ctxt is not None:
            ctxt = ctxt.unsqueeze(1)
            x = torch.cat((cls_tokens, x, c, ctxt), 1)
        else:
            x = torch.cat((cls_tokens, x, l, c), 1)
        x = x + self.pos_embedding
        x = self.pos_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, t_input_dim, nheads, dropout):
        super(Attention, self).__init__()
        dim_head = t_input_dim // nheads
        inner_dim = dim_head * nheads
        project_out = not (nheads == 1 and dim_head == t_input_dim)

        self.heads = nheads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(t_input_dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, t_input_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots).to(q.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        """

        b, n, _, _ = x.shape
        h = self.heads
        d = x.shape[-1] // h
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, d).transpose(1, 2) for t in qkv]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = self.attend(dots).to(q.dtype)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.patch_dim_d = in_channels * 2 * self.patch_size ** 2
        self.patch_dim_v = in_channels * 2 * self.patch_size ** 2
        self.proj_d = nn.Linear(self.patch_dim_d, 512)
        self.proj_v = nn.Linear(self.patch_dim_v, 64)

    def forward(self, d, v):
        """
        d = rearrange(d, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        v = rearrange(v, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        d = self.proj_d(d) #[b, 1, embed_dim]
        v = self.proj_v(v)  # [b, 1, embed_dim]
        x = torch.cat((d, v), dim=2)
        """

        b, c, h, w = d.shape
        p1 = p2 = self.patch_size

        d = d.view(b, c, h // p1, p1, w // p2, p2).permute(0, 2, 4, 3, 5, 1).contiguous().view(b, h * w, p1 * p2 * c)
        v = v.view(b, c, h // p1, p1, w // p2, p2).permute(0, 2, 4, 3, 5, 1).contiguous().view(b, h * w, p1 * p2 * c)

        d = self.proj_d(d) #[b, 1, embed_dim]
        v = self.proj_v(v)  # [b, 1, embed_dim]
        x = torch.cat((d, v), dim=2)
        
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)