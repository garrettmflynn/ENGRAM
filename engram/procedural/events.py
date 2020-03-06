
def select(name,reader):
    selection = {
        "RAM": RAM,
        "Neurogenesis":Neurogenesis
    }
    # Get the function from switcher dictionary
    func = selection.get(name, lambda: "Invalid event parser")
    # Execute the function
    return func(reader)



def RAM(reader):
    reader.parse_header()
    nb_event_channel = reader.event_channels_count()

    events = {}
    neurons ={}

    for chan_index in range(nb_event_channel):
        nb_channel = reader.header['event_channels'][chan_index][0];
        event_name = None
        if 'nr' not in nb_channel:
            a = ['DIO_00002','DIO_65533']
            if any(x in nb_channel for x in a):
                event_name = 'ITI_ON'
            if any(x in nb_channel for x in ['DIO_00004','DIO_65531']):
                event_name = 'FOCUS_ON'
            if any(x in nb_channel for x in ['DIO_00008','DIO_65527']):
                event_name = 'SAMPLE_ON'
            if any(x in nb_channel for x in ['DIO_00016','DIO_65519']):
                event_name = 'SAMPLE_RESPONSE'
            if any(x in nb_channel for x in ['DIO_00032','DIO_65503']):
                event_name = 'MATCH_ON'
            if any(x in nb_channel for x in ['DIO_00064','DIO_65471']):
                event_name = 'MATCH_RESPONSE'
            if any(x in nb_channel for x in ['DIO_00128','DIO_65407']):
                event_name = 'CORRECT_RESPONSE'
            if any(x in nb_channel for x in ['DIO_00256','DIO_65279']):
                event_name = 'END_SESSION'
            if any(x in nb_channel for x in ['DIO_Changed','DIO_65533']):
                event_name = 'DIO_CHANGED'

            times = reader.get_event_timestamps(block_index=0, seg_index=0,
                                                event_channel_index=chan_index,
                                                t_start=None,
                                                t_stop=None)
            times = reader.rescale_event_timestamp(times[0], dtype='float64')
            events[event_name] = times

        else:
            neuron_name = nb_channel
            times = reader.get_event_timestamps(block_index=0, seg_index=0,
                                                event_channel_index=chan_index,
                                                t_start=None,
                                                t_stop=None)
            times = reader.rescale_event_timestamp(times[0], dtype='float64')
            neurons[neuron_name] = times 

    print('Done parsing RAM events')

    return events,neurons


def Neurogenesis():
    print('Second option. Placeholder only')
