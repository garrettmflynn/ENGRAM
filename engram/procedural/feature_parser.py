

def select(name,settings):
    selection = {
        "LFP": LFP,
        "STFT": STFT
    }
    # Get the function from switcher dictionary
    func = selection.get(name, lambda: "Invalid event parser")
    # Execute the function
    return func(settings)



def LFP():
    print('First option. Placeholder only')


def STFT():
    print('Second option. Placeholder only')
