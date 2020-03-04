
def unpackNeo(reader):
    blks = reader.read(lazy=False)
    for blk in blks:
        for seg in blk.segments:
            raw_sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0)
            float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64')
            sampling_rate = reader.get_signal_sampling_rate()
            #t_start = reader.get_signal_t_start(block_index=0, seg_index=0)
            units = reader.header['signal_channels'][0]['units']

    return float_sigs,sampling_rate,units