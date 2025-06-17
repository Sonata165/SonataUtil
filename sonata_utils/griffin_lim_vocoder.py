import torch
import torch.nn as nn
import torchaudio


class GriffinLimVocoder(nn.Module):
    def __init__(self, stft_iter=1000, griflim_iter=30):
        super().__init__()

        sr = 8000
        n_fft = 800
        hop_length = 200
        n_mel = 80

        stft_spec_iter = stft_iter # Default: 100K
        griffin_lim_iter = griflim_iter

        f_max = sr // 2
        n_stft = int((n_fft//2) + 1)

        self.invers_transform = torchaudio.transforms.InverseMelScale(
            n_stft=n_stft,
            n_mels=n_mel,
            sample_rate=sr,
            f_max=f_max,
            max_iter=stft_spec_iter,
        ).cuda()

        self.grifflim_transform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=griffin_lim_iter,
            win_length=n_fft,
            hop_length=hop_length,
        ).cuda()

    def __call__(self, x):
        '''
        x: mel spectrogram to be reconstructed
        '''
        # self.train()

        # Patch for inverse stft
        x_ = torch.zeros_like(x, requires_grad=True)
        # x_.requires_grad = False
        x = x.float()
        with torch.enable_grad():
            stft_recon = self.invers_transform(x) # [bs, stft_bin, n_frame]

            # Pad last dim to >=2
            if stft_recon.shape[-1] < 5:
                pad_size = 5 - stft_recon.shape[-1]
                stft_recon = torch.nn.functional.pad(stft_recon, (0, pad_size), 'constant', 0)

            wav_recon = self.grifflim_transform(stft_recon)
        wav_recon = normalize_waveform_batch(wav_recon)
        wav_recon = wav_recon.unsqueeze(1)
        return wav_recon


def normalize_waveform_batch(waveform, target_db=-0.1):
    """
    Normalize the waveform to a specific dB level.

    Parameters:
    waveform (torch.Tensor): The input waveform tensor.
    target_db (float): The target dB level for normalization.

    Returns:
    torch.Tensor: The normalized waveform.
    """
    normalized_waveform = torch.zeros_like(waveform)
    for i in range(waveform.shape[0]):
        # Calculate the target amplitude
        target_amplitude = 10 ** (target_db / 20)

        # Find the peak amplitude in the waveform
        peak_amplitude = torch.max(torch.abs(waveform[i]))

        # Calculate the scaling factor
        scaling_factor = target_amplitude / peak_amplitude

        # Normalize the waveform
        normalized_waveform[i] = waveform[i] * scaling_factor

    return normalized_waveform
