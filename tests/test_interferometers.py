if __name__ == "__main__":

    from bosonicplus.interferometers.parameters import gen_interferometer_params
    params = gen_interferometer_params(10, 12)
    params = gen_interferometer_params(10, 12, bs_arrange='cascade')
    params = gen_interferometer_params(10, 12, bs_arrange='inv_cascade')

    from bosonicplus.interferometers.construct import build_interferometer
    circuit = build_interferometer(params, 10, True)