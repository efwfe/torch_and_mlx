# %% load package
import torchaudio
from plot_audio import plot_specgram, plot_waveform
import seaborn as sns
import matplotlib.pyplot as plt

# %% 
data_waveform, sr = torchaudio.load(wav_file)

# %%
data_waveform.size()

# %%
plot_waveform(data_waveform, sample_rate=sr)

# %% 
spectogram = torchaudio.transforms.Spectrogram()(data_waveform)
spectogram.size()

# %%
plot_specgram(waveform=data_waveform, sample_rate=sr)