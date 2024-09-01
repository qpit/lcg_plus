from bosonicplus.states.coherent import gen_fock_coherent, order_infidelity_fock_coherent
from bosonicplus.states.wigner import wig_mn
from bosonicplus.plotting import get_wigner_coherent_comb, plot_wig
from bosonicplus.quality.fidelity import fidelity_with_wigner
from matplotlib import pyplot as plt
import strawberryfields as sf
import numpy as np
from matplotlib.animation import ArtistAnimation, FuncAnimation
import os

def make_animation(N, save =True, xmax=6, xres=400, epsilons = np.linspace(0.5,4,30)):
    """Generate an animation of the coherent state approximation for Fock states.
    If save=True, automatically creates animations directory, and saves animation there.

    Args:
        N (int) : Fock number
        save (bool) : save the animation
        xmax (float) : x and p max
        xres (int) : resolution of wigner function
        epsilons (ndarray) : array of coherent state amplitudes to animate

    Returns:
        if save == True:
            saves animation in path/animations/Fock{N}.mp4
        else:
            ani (ArtistAnimation)
    """

    fig, ax = plt.subplots(figsize = (5,5))
    
    sf.hbar = 1
    x = np.sqrt(sf.hbar)*np.linspace(-xmax,xmax,xres)
    X, P = np.meshgrid(x,x )
    W_fock = wig_mn(N,N, X, P)
    
    ims = []
    for i, eps in enumerate(epsilons):
        data = gen_fock_coherent(N, 1, eps)
        covs, means, weights = data
    
        infid = order_infidelity_fock_coherent(N, eps)
        W = get_wigner_coherent_comb(data, x, x)
        im = plot_wig(ax, W, x, x, colorbar = False)
        
        Drawing_uncolored_circle = plt.Circle( (0,0 ), np.sqrt(2*sf.hbar)*eps ,fill = False, color = 'black' , linestyle='dashed',linewidth=2)
    
        theta = 2*np.pi/(N+1)
        betas = np.zeros(N+1, dtype='complex')
        for l in np.arange(N+1):
            betas[l] = eps * np.exp(1j * theta * l)
            
        im_gs = ax.scatter(np.sqrt(sf.hbar*2)*betas.real, np.sqrt(sf.hbar*2)*betas.imag, marker='o',color='k')
        
        #im_arrow = ax.arrow(0, 0, np.sqrt(2*sf.hbar)*eps, 0, color = 'red', linewidth = 2, head_width = 0.2, head_length = 0)
        im_eps = ax.text(2, 3.5, rf'$\epsilon = {np.round(eps,3)}$', size = 12)
         
        ax.set_aspect(1)
        im_circ = ax.add_artist(Drawing_uncolored_circle )
    
        fid = fidelity_with_wigner(W_fock, W, x, x).real
        ax.set_axis_off
        title = ax.text(0.25,1.01,r' $\mathcal{F}= $'+'{:.5f} with Fock {}'.format(fid,N),transform=ax.transAxes, fontsize =12 )
        ims.append([im, im_circ, im_gs, im_eps, title])
    
    ims.reverse() #From large to small eps
    ani = ArtistAnimation(fig, ims,interval = 150, blit = True,repeat=True)
    if save:
        if not os.path.exists('animations'):
            os.makedirs('animations')
        ani.save(f'animations/Fock{N}.mp4')
    if not save:
        return ArtistAnimation(fig, ims,interval = 150, blit = True,repeat=True)
