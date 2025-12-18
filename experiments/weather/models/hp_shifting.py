import torch
import numpy as np
import chealpix as chp


def get_attn_mask_from_mask(mask, window_size, window_partition):
    """Translates mask of shape (N) with different int values to attention mask of shape (nW,
    window_size, window_size) with values in {0,-100} suitable for attention module"""

    mask = mask[None, :, :, None]
    mask_windows = window_partition(mask, window_size)
    mask_windows = mask_windows.squeeze()

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class NoShift:
    def get_mask(self, window_partition):
        return None

    def shift(self, x):
        return x

    def shift_back(self, x):
        return x


class RingShift:
    def __init__(self, nside, base_pix, window_size, shift_size, input_resolution):
        self.nside = nside
        self.base_pix = base_pix
        self.npix = base_pix * self.nside**2
        self.ws = window_size
        self.shift_size = shift_size
        self.window_size_d, self.window_size_hp = window_size
        self.input_resolution = input_resolution
        self.shift_size_d = self.window_size_d // 2
        self.shift_size_hp = shift_size

        self.shift_idcs, self.mask = self._get_shifted_idcs_and_mask()
        self._validate_shift_result()
        self.back_shift_idcs = self._get_inverse_index_map(self.shift_idcs)

    def _get_shifted_idcs_and_mask(self):
        """shift hp image by converting to ring ordering and shifting there, then converting back to
        nest ordering

        """
        D, N = self.input_resolution

        ring_idcs = np.arange(12 * self.nside**2)
        # [0, 1, 2, 3, ...]
        shifted_ring_idcs = np.roll(ring_idcs, -self.shift_size_hp)
        # [2, 3, ..., 0, 1]
        shifted_ring_idcs_in_nest = chp.ring2nest(self.nside, shifted_ring_idcs)
        # [2_n, 3_n, ...,0_n, 1_n]

        # so far, this would return the image in ring indices, convert back to nested:
        nest_idcs = np.arange(self.npix)
        nest_idcs_in_ring = chp.nest2ring(self.nside, nest_idcs)
        result = shifted_ring_idcs_in_nest[nest_idcs_in_ring]

        mask = np.zeros((D, N))

        d_slices = (
            slice(-self.window_size_d, -self.shift_size_d),
            slice(-self.shift_size_d, None),
        )
        n_slices = (
            slice(0, -self.window_size_hp),
            slice(-self.window_size_hp, -self.shift_size_hp),
            slice(-self.shift_size_hp, None),
        )
        cnt = 0
        for d in range(0, D, self.window_size_d):
            for n in n_slices:
                for d_idx in range(self.window_size_d):
                    mask[d + d_idx, n] = cnt
                cnt += 1

        multiplier = cnt
        for d_idx, d in enumerate(d_slices):
            mask[d, :] += (d_idx + 1) * multiplier
        assert mask[-1, 0] != mask[-2, 0]
        assert mask[-1, -self.shift_size_hp - 1] != mask[-1, -1]

        for d_idx in range(D):
            mask[d_idx, :] = mask[d_idx, nest_idcs_in_ring]

        mask = torch.tensor(mask, dtype=torch.int64)
        result = torch.tensor(result, dtype=torch.int64)

        return result, mask

    def _validate_shift_result(self):
        diff = torch.max(torch.abs(self.shift_idcs.sort()[0] - torch.arange(self.npix)))
        assert (
            diff == 0
        ), f"shift validation failed for nside={self.nside}, window_size={self.ws}"

    def _get_inverse_index_map(self, idcs):
        return torch.sort(idcs)[1]

    def get_mask(self, window_partition, get_attn_mask=True):
        mask = self.mask
        if get_attn_mask:
            return get_attn_mask_from_mask(mask, self.ws, window_partition)
        else:
            return mask

    def shift(self, x):
        return torch.roll(
            x[:, :, self.shift_idcs, ...].contiguous(),
            shifts=[-self.window_size_d // 2],
            dims=[1],
        )

    def shift_back(self, x):
        return torch.roll(
            x[:, :, self.back_shift_idcs, ...].contiguous(),
            shifts=[self.window_size_d // 2],
            dims=[1],
        )
