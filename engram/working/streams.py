import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from engram.procedural.missingdata import interpolate_nans
from engram.working import loggers
from engram.procedural import features,predict
import tensorflow as tf

class DataManager(object):
    def __init__(self,source='SYNTHETIC',event_sources='KEYLOGGER',port=None):

        self.board = self.initialize(source,port)
        self.session_start = self.start()
        self.event_sources = EventManager(event_sources,self.session_start)
        print('Data is now being managed...')

    def initialize(self,source, port):

        if source in ['SYNTHETIC','OPENBCI']:
            if source == 'SYNTHETIC':
                BoardShim.enable_dev_board_logger()

                # use synthetic board for demo
                params = BrainFlowInputParams()
                board_id = BoardIds.SYNTHETIC_BOARD.value
                board = BoardShim(board_id, params)
                board.rate = BoardShim.get_sampling_rate(board_id)
                board.channels = BoardShim.get_eeg_channels(board_id)
                board.time_channel = BoardShim.get_timestamp_channel(board_id)
                board.eeg_channels = BoardShim.get_eeg_channels(board_id)
                board.accel_channels = BoardShim.get_accel_channels(board_id)

            elif source == 'OPENBCI':

                board_id = BoardIds.CYTON_DAISY_BOARD.value
                params = BrainFlowInputParams()
                params.serial_port = port
                board_id = BoardIds.CYTON_DAISY_BOARD.value
                board = BoardShim(board_id, params)
                board.rate = BoardShim.get_sampling_rate(board_id)
                board.channels = BoardShim.get_eeg_channels(board_id)
                board.time_channel = BoardShim.get_timestamp_channel(board_id)
                board.eeg_channels = BoardShim.get_eeg_channels(board_id)
                board.accel_channels = BoardShim.get_accel_channels(board_id)

            board.prepare_session()

        else:
            print('No stream with the specified name. Try "SYNTHETIC" or "OPENBCI" instead.')

        return board


    def start(self):
        self.board.start_stream(num_samples=450000)
        start_time = time.time()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
        return start_time


    def pull(self,num_samples=450000):
        data = self.board.get_current_board_data(num_samples=num_samples)
        for idx,values in enumerate(data):
            nans, x = interpolate_nans(values)
            values[nans] = np.interp(x(nans), x(~nans), values[~nans])
            data[idx] = values

        return data


    def stop(self):
        self.board.stop_stream()
        stream_time = time.time() - self.session_start
        self.board.release_session()

        return stream_time

    def predict(self,categories=None, settings=None):
        data = self.pull()
        t = data[self.board.time_channel]
        data = data[self.board.eeg_channels]
        t = t - t[0]
        settings['fs'] = len(data[0]) / (t[-1])
        model = tf.keras.models.load_model(settings['model'])
        # data = filters.butter_bandpass_filter(data, 59.0, 61.0, memory.details['rate'], order=5)
        feat, _, _, _ = features.stft(data=data, t=t, t_bin=settings['t_bin'], rate=settings['fs'],
                                                  lims=(settings['2D_min'], settings['2D_max']))
        prediction = predict.predict(model=settings['model'], feature=feat, categories=categories)

        return prediction


    def align(self,event_times=None, data_times=None, events=None):
        curr_timestamp = 1
        prev_timestamp = 0
        closest_data = np.zeros(event_times.size, dtype=int)
        for ii in range(event_times.size):
            while True:
                curr_diff = abs(event_times[ii] - data_times[curr_timestamp])
                prev_diff = abs(event_times[ii] - data_times[prev_timestamp - 1])
                if curr_diff < prev_diff:
                    curr_timestamp += 1
                    prev_timestamp += 1
                if curr_diff > prev_diff:
                    closest_data[ii] = int(curr_timestamp)
                    curr_timestamp += 1
                    prev_timestamp += 1
                    break

        new_events = np.empty(data_times.size, dtype=str)
        for ii in range(len(closest_data) - 1):
            this_event = closest_data[ii]
            next_event = closest_data[ii + 1]
            stop = int(np.floor(this_event + (next_event - this_event) / 2))
            if ii == 0:
                start = int(np.floor(this_event))
            else:
                prev_event = closest_data[ii - 1]
                start = int(np.floor(this_event + (prev_event - this_event) / 2))
            for jj in range(stop - start):
                new_events[start + jj] = events[ii]

        return new_events


class EventManager(object):

    def __init__(self, types='KEYLOGGER', stream_start=0):

        self.loggers = {}
        self.times = {}
        for type in types:
            self.loggers[type] = loggers.select(type,stream_start)
            self.times[type] = []


    def update(self):
        for type in self.loggers:
            self.loggers[type].manage()

    def pull(self):
        for type in self.loggers:
            self.logs[type] = self.loggers[type].log
            self.times[type] = self.loggers[type].log

        return self.logs[type],self.times[type]