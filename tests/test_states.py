if __name__ == "__main__":
    from bosonicplus.states.nongauss import prepare_gkp_coherent, prepare_fock_coherent
    gkp = prepare_gkp_coherent(10,'0')
    fock = prepare_fock_coherent(50)
    from bosonicplus.states.from_sf import prepare_cat_bosonic
    cat = prepare_cat_bosonic(3,0,0, True)