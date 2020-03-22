
import numpy as np 
from engram.procedural import filters
from scipy import signal

def select(name,trace,settings):
    selection = {
        "LFP": LFP,
        "STFT": STFT,
        "spikes": spikes,
        "multiscale": multiscale
    }
    # Get the function from switcher dictionary
    func = selection.get(name, lambda: "Invalid event parser")
    # Execute the function
    return func(trace,settings)


def multiscale(trace,settings):
    power, t, f = STFT(trace,settings)
    spikes,_,_ = spikes(trace,settings)
    signals = np.concatenate(power,spikes)
    
    return signals, t, f

def spikes(trace,settings):
    spikes = trace['spikes']
    t = np.arange(0,np.size(trace['Data'], 1))/settings['fs']
    f = None

    return spikes, t,f

def LFP(trace, settings):

    lfp = np.empty(trace['Data'].shape)
    if np.ndim(trace['Data']) == 2:
        for channel in range(np.size(trace['Data'], 0)):
            lfp[channel, :] = filters.select('bandpass',trace['Data'][channel, :],min=settings['bandpass_min'],
                                                max=settings['bandpass_max'],
                                                fs=settings['fs'],
                                                order=5)
        t = np.asarray(range(np.size(trace['Data'], 1)))/settings['fs']

    elif np.ndim(trace['Data']) == 1:
        lfp = filters.select('bandpass',trace['Data'],min=settings['bandpass_min'],
                                                max=settings['bandpass_max'],
                                                fs=settings['fs'],
                                                order=5)
        t = range(np.size(trace['Data'],1))/settings['fs']
        

    else:
        print('Input array has too many dimensions')

    f = None
    
    return lfp, t,f




def STFT(trace,settings):

    lfp,t,f = LFP(trace,settings)

    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * settings['fs'] / 2
    time = np.arange(N) / float(settings['fs'])
    mod = 500 * np.cos(2 * np.pi * 0.25 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    noise = np.random.normal(scale=np.sqrt(noise_power),
                                size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier + noise

    window= int(settings['t_bin'] * settings['fs'])
    c_len = lfp.shape[0]

    if np.ndim(lfp) == 2:
        for channel in range(len(lfp)):
            temp_f, temp_t, temp_Zxx = signal.spectrogram(lfp[channel, :],
                                                            settings['fs'],
                                                            'hann', nperseg=window)

            freq_slice = np.where((temp_f >= settings['2D_min']) & (temp_f <= settings['2D_max']))
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
                                                        settings['fs'],
                                                        'hann', nperseg=window)

        freq_slice = np.where((temp_f >= settings['2D_min']) & (temp_f <= settings['2D_max']))
        Zxx = temp_Zxx[freq_slice, :][0]
        del temp_Zxx
        power = np.empty([c_len, np.shape(Zxx)[1], np.shape(Zxx)[0]], dtype=float)
        f = temp_f[freq_slice]
        t = temp_t
        del temp_t
        del temp_f
        power = np.transpose(Zxx ** 2) # Time x Freq
        del Zxx

    if settings['norm']:
        power = normalize(power,settings)

    return power, t,f

def normalize(data,settings):
    if settings['log_transform']:
        data = 10*np.log10(data)
    if settings['norm_method'] == 'ZSCORE':
        if np.ndim(data) == 3:
            freqMu = np.mean(data,axis=1)
            freqSig = np.std(data,axis=1)

            for channel in range(len(data)):
                data[channel,:,:] = (data[channel] - freqMu[channel])/freqSig[channel]
        elif np.ndim(data) == 2 or 1:
            freqMu = np.mean(data,axis=0)
            freqSig = np.std(data,axis=0)
            data = (data - freqMu)/freqSig
        else:
            print('Input array dimensions not supported for normalization.')
    
    return data