#import libraries
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.modules.activation import MultiheadAttention
from torch import Tensor
from typing import Optional, Tuple, Dict

#Define my LSTM model
class simpleLSTM(nn.Module):
    def __init__(self):
        super(simpleLSTM, self).__init__()
        input_size=1
        hidden_size=32
        num_layers=1
        dropout_rate=0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True, bias=True)
        self.fc = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.ELU()

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        c0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.act(self.fc(h_n[-1]))
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

# Create new bidirectional LSTM
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        input_size=1
        hidden_size=32
        num_layers=1
        dropout_rate=0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64,1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        c0 = torch.zeros(self.num_layers * 2, x.batch_sizes[0], self.hidden_size).to(x.data.device)

        _, (h_n, c_n) = self.lstm(x, (h0, c0))
        #out = F.elu(cf[-1, :, :]) #try to escape local minima
        out = self.fc(h_n[-1])
        out = F.elu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out
    
#Define my CNN models
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        #create sequential block of convolution + maxpooling
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.maxpool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = nn.ELU()

    def forward(self, x):
        out=self.maxpool1(self.bn1(self.conv1(x)))
        out=self.maxpool2(self.bn2(self.conv2(out)))
        out=self.maxpool3(self.bn3(self.conv3(out)))
        out=out.view(out.size(0), -1)
        out=self.act(self.fc1(out))
        out=self.fc2(out)
        out = F.sigmoid(out)
        return out
    
#Define my CNN model with residual connections

#Define a residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()
        
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.act(out)
        return out

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        #Use resblocks
        self.conv1 = nn.Conv1d(1, 24, 5, padding=1, stride=3)
        self.bn1 = nn.BatchNorm1d(24)
        self.resblock1 = ResBlock(24, 24, 3, 1, 1)
        self.resblock2 = ResBlock(24, 24, 3, 1, 1)
        self.resblock3 = ResBlock(24, 24, 3, 1, 1)
        self.maxpoolf = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(744, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = nn.ELU()

    def forward(self, x):
        out=self.bn1(self.conv1(x))
        out=self.resblock3(self.resblock2(self.resblock1(out)))
        out=self.maxpoolf(out)
        out=out.view(out.size(0), -1)
        out=self.act(self.fc1(out))
        out=self.fc2(out)
        out = F.sigmoid(out)
        return out
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 187):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)/187
        self.register_buffer('pe', position)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size,1embedding_dim,seq_len]``
        """
        xclone=x.clone()
        xclone[:, 1, :] = self.pe.squeeze().expand_as(xclone[:, 1, :])
        return xclone
    

#Define my model with attention, not recurrent
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.positional = PositionalEncoding(2)
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2, nhead=1,norm_first=True,batch_first=True,dim_feedforward=8), num_layers=2,enable_nested_tensor=False)
        self.linear2 = nn.Linear(2*187,128)
        self.linear3 = nn.Linear(128,1)
        self.act = nn.ELU()
    def forward(self, x):
        out = x.expand(-1, 2 , -1)
        out=self.positional(out)
        out=out.permute(0,2,1)
        out=self.TransformerEncoder(out)
        out=out.reshape(out.size(0),-1)
        out=self.act(self.linear2(out))
        out=self.linear3(out)
        out = F.sigmoid(out)
        return out

#support func
def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

#support func      
def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

#create transformer with attention weights outputted togheter, tuple[tensor,tensor]
class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    #initialize my class exactly as the parent class
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 norm_first=False, batch_first=False, bias=True, **factory_kwargs):
        super(MyTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward=dim_feedforward,
                                                         dropout=dropout, activation=activation,
                                                         norm_first=norm_first, batch_first=batch_first,
                                                         bias=bias, **factory_kwargs)
        
    #redefine the _sa_block
    def my_sa_block(self, x: Tensor,
                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                is_causal: bool = False) -> Tuple[Tensor, Tensor]:
        x, attention_weights = self.self_attn(x, x, x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        need_weights=True, is_causal=is_causal)
        return self.dropout1(x), attention_weights
    
    #redefine the forward method to also output the attention_weights
    def forward(self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False) -> Tuple[Tensor, Tensor]:

        src_key_padding_mask = F._canonical_mask(
        mask=src_key_padding_mask,
        mask_name="src_key_padding_mask",
        other_type=F._none_or_dtype(src_mask),
        other_name="src_mask",
        target_type=src[0].dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                            f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                            "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )


        x = src
        if self.norm_first:
            _sa_block_output=self.my_sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + _sa_block_output[0]
            x = x + self._ff_block(self.norm2(x))
        else:
            _sa_block_output=self.my_sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self.norm1(x + _sa_block_output[0])
            x = self.norm2(x + self._ff_block(x))

        return x, _sa_block_output[1]

#
class MyTransformerEncoder(nn.TransformerEncoder):
    #initialize my class exactly as the parent class
    def __init__(self, encoder_layer, num_layers, enable_nested_tensor=False,norm=None):
        super(MyTransformerEncoder, self).__init__(encoder_layer, num_layers, norm)

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        attention_weights = {}
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        i=1
        for mod in self.layers:
            output, weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attention_weights[f'layer_{i}'] = weights
            i+=1
            
        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_weights


#Define my model with attention
class TransformerWithAttentionOutputted(Transformer):
    def __init__(self):
        super(TransformerWithAttentionOutputted, self).__init__()
        #change the transformerencoderlayer
        self.TransformerEncoder = MyTransformerEncoder(
            MyTransformerEncoderLayer(d_model=2, nhead=1,dim_feedforward=8,norm_first=True,batch_first=True),
            num_layers=2,enable_nested_tensor=False)
        
        
    def forward(self, x):
        out = x.expand(-1, 2 , -1)
        out=self.positional(out)
        out=out.permute(0,2,1)
        out, attention_weights=self.TransformerEncoder(out)
        out=out.reshape(out.size(0),-1)
        out=self.act(self.linear2(out))
        out=self.linear3(out)
        out = F.sigmoid(out)
        return out, attention_weights