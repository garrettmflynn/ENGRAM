'''
This module defines :class:`Cont`,  a container for continuous data.
'''
from engram.procedural import filters

import numpy as np
from scipy import signal

class Cont(object):
    def __init__(self, id, data=[], timestamps = [], \
                channel_labels = [], metadata=None):

        # Get number of elements in data
        numel = 1
        for dim in np.shape(data): numel *= dim

        # Check if data is continuous or binary
        if ((data==0) | (data==1)).all():
            return "Invalid continuous input. Nothing has been stored."
        else:
            self.id = id
            self.timestamps = np.asarray(timestamps)
            self.data = np.asarray(data) # Channels x Time
            self.representation = 'raw'
            self.metadata = metadata

            self.nD_labels = {'1D': np.asarray(channel_labels)}
            self.nD_labels['2D'] = np.arange(0,np.size(self.data, 1))/self.metadata['fs'] # Time
            self.nD_labels['3D'] = None # ???

    def __repr__(self):
        return "Cont('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)

    def lfp(self):

        lfp = np.empty(self.data.shape)
        if np.ndim(self.data) == 2:
            for channel in range(np.size(self.data, 0)):
                self.data[channel, :] = filters.select('bandpass',self.data[channel, :],min=self.metadata['bandpass_min'],
                                                    max=self.metadata['bandpass_max'],
                                                    fs=self.metadata['fs'],
                                                    order=5)
            self.nD_labels['2D'] = np.asarray(range(np.size(self.data, 1)))/self.metadata['fs']

        elif np.ndim(self.data) == 1:
            self.data = filters.select('bandpass',self.data,min=self.metadata['bandpass_min'],
                                                    max=self.metadata['bandpass_max'],
                                                    fs=self.metadata['fs'],
                                                    order=5)
            self.nD_labels['2D'] = range(np.size(self.data,1))/self.metadata['fs']
            

        else:
            print('Input array has too many dimensions')

        self.nD_labels['3D'] = None
        
        return self

    def stft(self):

        lfp = self.lfp().data

        N = 1e5
        amp = 2 * np.sqrt(2)
        noise_power = 0.01 * self.metadata['fs'] / 2
        time = np.arange(N) / float(self.metadata['fs'])
        mod = 500 * np.cos(2 * np.pi * 0.25 * time)
        carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
        noise = np.random.normal(scale=np.sqrt(noise_power),
                                    size=time.shape)
        noise *= np.exp(-time / 5)
        x = carrier + noise

        window= int(self.metadata['t_bin'] * self.metadata['fs'])
        c_len = lfp.shape[0]

        if np.ndim(lfp) == 2:
            for channel in range(len(lfp)):
                temp_f, temp_t, temp_Zxx = signal.spectrogram(lfp[channel, :],
                                                                self.metadata['fs'],
                                                                'hann', nperseg=window)

                freq_slice = np.where((temp_f >= self.metadata['2D_min']) & (temp_f <= self.metadata['2D_max']))
                Zxx = temp_Zxx[freq_slice, :][0]
                del temp_Zxx

                if channel == 0:
                    power = np.empty([c_len, np.shape(Zxx)[1], np.shape(Zxx)[0]], dtype=float)
                    f = temp_f[freq_slice]
                    t = temp_t
                del temp_t
                del temp_f
                power[channel, :, :] = np.transpose(Zxx ** 2) # Channels x Time x Freq
                del Zxx

        if np.ndim(lfp) == 1:
            temp_f, temp_t, temp_Zxx = signal.spectrogram(lfp,
                                                            self.metadata['fs'],
                                                            'hann', nperseg=window)

            freq_slice = np.where((temp_f >= self.metadata['2D_min']) & (temp_f <= self.metadata['2D_max']))
            Zxx = temp_Zxx[freq_slice, :][0]
            del temp_Zxx
            power = np.empty([c_len, np.shape(Zxx)[1], np.shape(Zxx)[0]], dtype=float)
            f = temp_f[freq_slice]
            t = temp_t
            del temp_t
            del temp_f
            power = np.transpose(Zxx ** 2) # Time x Freq
            del Zxx

        self.data = power
        self.nD_labels['2D'] = t
        self.nD_labels['3D'] = f

        if self.metadata['norm']:
            self.normalize()

        return self

    def normalize(self):
        if self.metadata['log_transform']:
            self.data = 10*np.log10(self.data)
        if self.metadata['norm_method'] == 'ZSCORE':
            if np.ndim(self.data) == 3:
                freqMu = np.mean(self.data,axis=1)
                freqSig = np.std(self.data,axis=1)

                for channel in range(len(self.data)):
                    self.data[channel,:,:] = (self.data[channel] - freqMu[channel])/freqSig[channel]
            elif np.ndim(self.data) == 2:
                freqMu = np.mean(self.data,axis=0)
                freqSig = np.std(self.data,axis=0)
                self.data = (self.data - freqMu)/freqSig
            else:
                print('Input array dimensions not supported for normalization.')

        return self

    def resample(self,fs):
        if fs == self.metadata['fs']:
            return 'Current frequency is already at the desired value.'
            
        else:
            num_points = round(np.shape(self.data,1) * (self.metadata['fs']/fs))
            self.data = signal.resample(self.data,num_points)
            data = filters.select('bandpass', min=0, max=self.metadata['fs'],
                                    fs=fs, order=5)
            downsample = np.shape(self.data,1)/np.shape(self.data,1)
            print('Sampled from ' + fs + ' to ' + downsample + 'Hz')
            fs = downsample