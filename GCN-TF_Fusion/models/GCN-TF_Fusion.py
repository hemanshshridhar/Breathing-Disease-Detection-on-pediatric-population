import torch
import math
import argparse

from contextlib import nullcontext


def wrapperkwargs(func, kwargs):
    return func(**kwargs)


def wrapperargs(func, args):
    return func(*args)



"""
Temporal FiLM layer
"""


class TFiLM(torch.nn.Module):
    def __init__(self,
                 nchannels,
                 block_size=128):
        super(TFiLM, self).__init__()
        self.nchannels = nchannels
        self.block_size = block_size
        self.num_layers = 1
        self.hidden_state = None  # (hidden_state, cell_state)

        # used to downsample input
        self.maxpool = torch.nn.MaxPool1d(kernel_size=block_size,
                                          stride=None,
                                          padding=0,
                                          dilation=1,
                                          return_indices=False,
                                          ceil_mode=False)

        self.lstm = torch.nn.LSTM(input_size=nchannels,
                                  hidden_size=nchannels,
                                  num_layers=self.num_layers,
                                  batch_first=False,
                                  bidirectional=False)

    def forward(self, x):
        # print("TFiLM: ", x.shape)
        # x = [batch, channels, length]
        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)
        device = x.device
        # downsample
        x_down = self.maxpool(x)

        # shape for LSTM (length, batch, channels)
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state == None:  # state was reset
            # init hidden and cell states with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.nchannels,device = device).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.nchannels,device = device).requires_grad_()
            x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back (batch, channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        x_in = torch.reshape(
            x, shape=(-1, self.nchannels, nsteps, self.block_size))
        x_norm = torch.reshape(
            x_norm, shape=(-1, self.nchannels, nsteps, 1))

        # multiply
        x_out = x_norm * x_in

        # return to original shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        return x_out

    def detach_state(self):
        if self.hidden_state.__class__ == tuple:
            self.hidden_state = tuple([h.clone().detach() for h in self.hidden_state])
        else:
            self.hidden_state = self.hidden_state.clone().detach()

    def reset_state(self):
        self.hidden_state = None
class GatedConv1d(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 kernel_size,
                 tfilm_block_size):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernal_size = kernel_size
        self.tfilm_block_size = tfilm_block_size

        # Layers: Conv1D -> Activations -> TFiLM -> Mix + Residual

        self.conv = torch.nn.Conv1d(in_channels=in_ch,
                                    out_channels=out_ch * 2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0,
                                    dilation=dilation)

        self.tfilm = TFiLM(nchannels=out_ch,
                           block_size=tfilm_block_size)

        self.mix = torch.nn.Conv1d(in_channels=out_ch,
                                   out_channels=out_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x):
        # print("GatedConv1d: ", x.shape)
        residual = x

        # dilated conv
        y = self.conv(x)

        # gated activation
        z = torch.tanh(y[:, :self.out_ch, :]) * \
            torch.sigmoid(y[:, self.out_ch:, :])
        device = x.device
        # zero pad on the left side, so that z is the same length as x
        z = torch.cat((torch.zeros(residual.shape[0],self.out_ch,residual.shape[2] - z.shape[2],device= device),z),dim=2)

        # modulation
        z = self.tfilm(z)

        x = self.mix(z) + residual

        return x, z
 class GCNBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 nlayers,
                 kernel_size,
                 dilation_growth,
                 tfilm_block_size):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size

        dilations = [dilation_growth ** l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()

        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch,
                                           out_ch=out_ch,
                                           dilation=d,
                                           kernel_size=kernel_size,
                                           tfilm_block_size=tfilm_block_size))
            in_ch = out_ch

    def forward(self, x):
        # print("GCNBlock: ", x.shape)
        # [batch, channels, length]
        z = torch.empty([x.shape[0],
                         self.nlayers * self.out_ch,
                         x.shape[2]])

        for n, layer in enumerate(self.layers):
            x, zn = layer(x)
            z[:, n * self.out_ch: (n + 1) * self.out_ch, :] = zn

        return x, z     
class GCNTF1(torch.nn.Module):
    def __init__(self,
                 nblocks=3,
                 nlayers=2,
                 nchannels=128,
                 kernel_size=3,
                 dilation_growth=2,
                 tfilm_block_size=133,
                 **kwargs):
        super(GCNTF1, self).__init__()
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch= nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        tfilm_block_size=tfilm_block_size))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x):
        # print("GCN: ", x.shape)
        # x.shape = [length, batch, channels]
        device = x.device
        x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]],device = device)
        
        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)
  
            z[:,n * self.nchannels * self.nlayers:(n + 1) * self.nchannels * self.nlayers,:] = zn
             

        return self.blocks[-1](z).permute(2, 0, 1)

    def detach_states(self):
        # print("DETACH STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.detach_state()

    # reset state for all TFiLM layers
    def reset_states(self):
        # print("RESET STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()
    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf            
class GCNTF2(torch.nn.Module):
    def __init__(self,
                 nblocks=3,
                 nlayers=4,
                 nchannels=128,
                 kernel_size=3,
                 dilation_growth=2,
                 tfilm_block_size=133,
                 **kwargs):
        super(GCNTF2, self).__init__()
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch= nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        tfilm_block_size=tfilm_block_size))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x):
        # print("GCN: ", x.shape)
        # x.shape = [length, batch, channels]
        device = x.device
        x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]],device = device)
        
        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)

            z[:,n * self.nchannels * self.nlayers:(n + 1) * self.nchannels * self.nlayers,:] = zn
             

        return self.blocks[-1](z).permute(2, 0, 1)

    def detach_states(self):
        # print("DETACH STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.detach_state()

    # reset state for all TFiLM layers
    def reset_states(self):
        # print("RESET STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()
    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf            
class GCNTF3(torch.nn.Module):
    def __init__(self,
                 nblocks=3,
                 nlayers=6,
                 nchannels=128,
                 kernel_size=3,
                 dilation_growth=2,
                 tfilm_block_size=133,
                 **kwargs):
        super(GCNTF3, self).__init__()
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch= nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        tfilm_block_size=tfilm_block_size))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x):
        # print("GCN: ", x.shape)
        # x.shape = [length, batch, channels]
        device = x.device
        x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]],device = device)
        
        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)

            z[:,n * self.nchannels * self.nlayers:(n + 1) * self.nchannels * self.nlayers,:] = zn
             
 

        return self.blocks[-1](z).permute(2, 0, 1)

    def detach_states(self):
        # print("DETACH STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.detach_state()

    # reset state for all TFiLM layers
    def reset_states(self):
        # print("RESET STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()
    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf            
class GCNTF_FUSION(torch.nn.Module):
    def __init__(self):
        super(GCNTF_FUSION,self).__init__()
        self.model1 = GCNTF1()
        self.model2 = GCNTF2()
        self.model3 = GCNTF3()
    def forward(self,x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x1 = x1.permute(1,2,0)
        x2 = x2.permute(1,2,0)
        x3 = x3.permute(1,2,0)
        y = torch.cat([x1,x2,x3])
        
        return y
        
