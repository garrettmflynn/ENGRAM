
import numpy as np
def select(feature,time,settings,prev_len = None):
    selection = {
        "roi": roi,
        "trials": trials
    }
    # Get the function from switcher dictionary
    func = selection.get(settings['mneme_method'], lambda: "Invalid event parser")
    # Execute the function
    return func(feature,time,settings,prev_len)


def roi(feature,time,settings,prev_len):
    bounds = settings['roi_bounds']
    upper_index = (np.abs(settings['t_feat'] - (time + bounds[1]))).argmin()
    lower_index = (np.abs(settings['t_feat'] - (time + bounds[0]))).argmin()
    diff_one = upper_index-lower_index
    if prev_len:
        if prev_len != diff_one:
            diff_two = prev_len - diff_one
            if diff_one % 2:
                center = lower_index+diff_one/2
                offset = (prev_len+1)/2
                upper_index = center + offset
                lower_index = center - offset
                upper_index += diff_two
            else:
                upper_index += diff_two

    section = feature[lower_index:upper_index]
    prev_len = upper_index-lower_index
    return section,prev_len

def trials():
    print('trial segmentation not available')