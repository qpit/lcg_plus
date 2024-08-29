from strawberryfields.backends.states import BaseBosonicState

def get_state(circuit):
    """Get BaseBosonic state object from BosonicModes object
    """
    data = circuit.means, circuit.covs, circuit.weights
    num_modes = len(circuit.get_modes())
    state = BaseBosonicState(data, num_modes, len(data[2]))
    return state