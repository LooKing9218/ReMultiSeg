import math
import torch
import torch.nn as nn
import torch.nn.functional as F


HORIZONTAL_FIRST = True

class Splitting(nn.Module):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        self.horizontal = horizontal
        if(horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        return (self.conv_even(x), self.conv_odd(x))






class LiftingScheme(nn.Module):
    def __init__(self, horizontal, in_planes, modified=True, splitting=True,
                 k_size=3):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        if horizontal:
            kernel_size = (1, k_size)
            pad = (k_size // 2, k_size - 1 - k_size // 2, 0, 0)
        else:
            kernel_size = (k_size, 1)
            pad = (0, 0, k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(horizontal)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        size_hidden = 2
        modules_P += [
            nn.ZeroPad2d(pad),
            nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                      kernel_size=kernel_size, stride=1),
            nn.ReLU()
        ]
        modules_U += [
            nn.ZeroPad2d(pad),
            nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                      kernel_size=kernel_size, stride=1),
            nn.ReLU()
        ]
        prev_size = size_hidden

        # Final dense
        modules_P += [
            nn.Conv2d(in_planes*prev_size, in_planes,
                      kernel_size=(1, 1), stride=1),
            nn.Tanh()
        ]
        modules_U += [
            nn.Conv2d(in_planes*prev_size, in_planes,
                      kernel_size=(1, 1), stride=1),
            nn.Tanh()
        ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        (x_even, x_odd) = self.split(x)
        c = x_even + self.U(x_odd)
        d = x_odd - self.P(c)
        return (c, d)



class LiftingScheme2D(nn.Module):
    def __init__(self, in_planes, modified=True, kernel_size=3):
        super(LiftingScheme2D, self).__init__()
        print("============ LiftingScheme2D V4 ============")
        self.level1_lf = LiftingScheme(
            horizontal=HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
            k_size=kernel_size)
        self.level2_1_lf = LiftingScheme(
            horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
            k_size=kernel_size)
        self.level2_2_lf = LiftingScheme(
            horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
            k_size=kernel_size)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.level1_lf(x)
        (LL, LH) = self.level2_1_lf(c)
        (HL, HH) = self.level2_2_lf(d)
        return (c, d, LL, LH, HL, HH)


if __name__ == "__main__":
    input = torch.randn(1, 1, 10, 10)
    #m_harr = WaveletLiftingHaar2D()
    m_wavelet = LiftingScheme2D(1, name="db2")
    print(input)
    print(m_wavelet(input))

    # TODO: Do more experiments with the code
