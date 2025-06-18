import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from mpmath import mp, fp
hbar = 2
from bosonicplus.sampling import *


def plot_wig(W, q1, q2, colorbar = True, xlabel = None, ylabel = None, GKP = False, ax = None):
    if not ax:
        ax_ = plt.gca()
        
    #W = np.round(W.real, 4)
    scale = np.max(W.real)
    nrm = mpl.colors.Normalize(-scale, scale)
    if GKP:
        im = plt.contourf(q1 /np.sqrt(hbar * np.pi), q2 /np.sqrt(hbar * np.pi), W, 100, cmap=cm.RdBu, norm = nrm)
        plt.xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
        plt.ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
        plt.grid('on')
    else:
        
        im = plt.contourf(q1, q2, W, 100, cmap=cm.RdBu, norm = nrm)
        
        #im = plt.imshow(q1, q2, W, 100, cmap='RdBu', norm = nrm)
        
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
        else:
            plt.xlabel(r'$x$', fontsize=12)
            plt.ylabel(r'$p$', fontsize=12)
    
    #ax.set_xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
    #ax.set_ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
    
    if colorbar:
        plt.colorbar(cm.ScalarMappable(norm = nrm, cmap = cm.RdBu), ax=ax_, shrink = 0.82)
    
    #ax_.set_rasterized(True)
    #ax_.set_rasterization_zorder(0)
    #ax_.set_aspect("equal")
    
    return im

def make_grid(ax, x, p, axis = 'both'):
    r"""make custom grid to avoid pixel snapping in imshow.
    """
    kwargs = {'color':'gray', 'linewidth' :0.2, 'alpha': 0.5}
    xmax = np.max(x)
    xmin = np.min(x)
    pmax = np.max(p)
    pmin = np.min(p)

    if axis == 'x' or axis == 'both':
        
        x_grid = np.floor([xmin,xmax])
        
        for n in np.arange(x_grid[0],x_grid[1]+1):
            ax.vlines(n, pmin, pmax, **kwargs)    

    ax.set_xlim([xmin,xmax])
        
    if axis == 'p' or axis == 'both':
        
        p_grid = np.floor([pmin,pmax])
        
        for n in np.arange(p_grid[0],p_grid[1]+1):
            
            ax.hlines(n, xmin, xmax, **kwargs)

    ax.set_ylim([pmin,pmax])

def plot_wigner_marginals(W, x, p, **kwargs):
    """Plot the Wigner function, including marginals and colorbar.
    """

    if kwargs:
        title = kwargs['title']
        fs = kwargs['fontsize']
        figsize = kwargs['figsize']
        xp_grid = kwargs['grid'] 
        lw = kwargs['linewidth']
        xlim = kwargs['xlim']
        plim = kwargs['plim']
       
    else:
        title = None
        fs = 10
        figsize = (4,4)
        xp_grid = None
        xlim = np.max(x)
        plim = np.max(p)
        lw = 1

        
    
    fig = plt.figure(figsize=figsize)
    
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal Axes and the main Axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    #gs = fig.add_gridspec(2,2,  width_ratios=(1,4), height_ratios=(1,4),
                          #left=0.15, right=0.85, bottom=0.15, top=0.85,
                          #wspace=0.15, hspace=0.15)

    gs = fig.add_gridspec(2,3,  width_ratios=(1,4,0.25), height_ratios=(1,4),
                          left=0.15, right=0.85, bottom=0.1, top=0.75,
                          wspace=0.25, hspace=0.25, )
    #gs = fig.add_gridspec(2,3,  width_ratios=(1,4,0.25), height_ratios=(1,4),
                          #wspace=0.15, hspace=0.15, )
    
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[0, 1], sharex = ax)
    ax_p = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1,2])


    marginal_x = np.sum(W,axis=0)*np.diff(p)[0]
    marginal_p = np.sum(W,axis=1)*np.diff(x)[0]

    
    W = W.real
    scale = np.max(W)
    nrm = mpl.colors.Normalize(-scale, scale)
    extent = np.array([np.min(x), np.max(x), np.min(p), np.max(p)])

    #Get scaling and labels

    if xp_grid == None:
        gridx = 1.0
        gridp = 1.0
        ax.set_xlabel(r"$x$")
        ax_p.set_ylabel(r"$p$")

    elif xp_grid == 'rect':
        gridx = np.sqrt(hbar*np.pi)
        gridp = np.sqrt(hbar*np.pi)
        ax.set_xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$")
        ax_p.set_ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$")
        
    elif xp_grid =='square':
        gridx = np.sqrt(2*hbar*np.pi)
        gridp = np.sqrt(2*hbar*np.pi)
        ax.set_xlabel(r"$x(\sqrt{2\pi\hbar})^{-1}$")
        ax_p.set_ylabel(r"$p(\sqrt{2\pi\hbar})^{-1}$")
        
    elif xp_grid == 'hex':
        gridx = (3/4)**(1/4) * np.sqrt(hbar*np.pi)
        gridp = (4/3)**(1/4) * np.sqrt(hbar*np.pi)
        ax.set_xlabel(r"$x(\sqrt{\frac{\sqrt{3}}{2}\pi\hbar})^{-1}$")
        ax_p.set_ylabel(r"$p(\sqrt{\frac{2}{\sqrt{3}}\pi\hbar})^{-1}$")

    elif xp_grid == 'hex_square':
        gridx = (3/4)**(1/4) * np.sqrt(2*hbar*np.pi)
        gridp = (4/3)**(1/4) * np.sqrt(2*hbar*np.pi)
        
        ax.set_xlabel(r"$x(\sqrt{\sqrt{3}\pi\hbar})^{-1}$")
        ax_p.set_ylabel(r"$p(2\sqrt{\frac{1}{\sqrt{3}}\pi\hbar})^{-1}$")
        
    

    #Make grid for Wigner function and plot it 
    make_grid(ax, x/gridx, p/gridp)
    extent[0:2]/=gridx
    extent[2:]/=gridp
    im = ax.imshow(W, cmap='RdBu', norm = nrm, extent = extent, interpolation = 'bilinear')
    #ax_p.xaxis.set_inverted(True) 
    ax.set_aspect("equal")
    

    
    #Make the grid for the marginals
    make_grid(ax_x, x/gridx, marginal_x, axis='x')
    make_grid(ax_p, marginal_p, p/gridp, axis='p')

    #Plot the marginals
    ax_x.plot(x/gridx, marginal_x, linewidth = lw)
    ax_p.plot(marginal_p, p/gridp, linewidth = lw)
    
    ax_x.tick_params(axis = 'x',labelbottom = False)
    #ax_p.set_xticks([0, np.round(np.max(marginal_p),2)])
    #ax_x.set_yticks([0, np.round(np.max(marginal_x),2)])

    ax.set_xlim([-xlim,xlim])
    ax.set_ylim([-plim,plim])
    ax_x.set_ylim([0, np.max(marginal_x)])
    ax_p.set_xlim([0, np.max(marginal_p)])
    ax_p.set_ylim([-xlim,xlim])
    #ax_x.set_
    #ax_x.set_yticks([np.min(marginal_x), np.max(marginal_x)])
    #ax_x.set_yticks([])
    #ax_p.set_xticks([])
    #ax_p.xaxis.set_major_locator(plt.MaxNLocator(0))
    
    ax.tick_params(axis = 'y', labelleft=False)

    
    ax_x.set_ylabel(r'$P(x)$')
    ax_p.set_xlabel(r'$P(p)$')
    ax_p.invert_xaxis()
    ax_p.invert_yaxis()

    plt.colorbar(im, cax = cax )
    #if title:
        #fig.suptitle(title, horizontalalignment = kwargs['title_loc'], bbox=(-1))
    #fig.tight_layout()
    
    return fig, ax, ax_x, ax_p, cax

def make_joint_distribution_plot(state, x):
    if state.num_modes != 2: 
        raise ValueError('incompatible when number of modes != 2')
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))

    for ax_, idx_, xlab_, ylab_ in [(ax1, [0,1],r'$x_1$',r'$p_1$'), 
                              (ax2,[0,2],r'$x_1$',r'$x_2$'),
                              (ax3,[1,3],r'$p_1$',r'$p_2$'),
                              (ax4,[2,3],r'$x_2$',r'$p_2$')]:
        plt.sca(ax_)
        W = state.get_wigner_bosonic(x,x, idx_)
        im = plot_wig(W, x,x,colorbar=False, xlabel=xlab_, ylabel=ylab_)

    plt.tight_layout()
    



def Gauss(sigma, mu, x, p, MP = False):
    """
    Returns the Gaussian in phase space point (x,p), or on a grid

    Args:
        sigma : covariance matrix
        mu : displacement vector
        x : np.ndarray 
        p : np.ndarray
        MP : bool
    Works for one sigma and one mu
    """

    if len(p)==1:
        xi  = x

    else:

        X, P = np.meshgrid(x,p)
    
        xi = np.array([X,P])

    sigma_inv = np.linalg.inv(sigma)

    delta = xi - mu[:,np.newaxis, np.newaxis]

    sigma_inv = np.linalg.inv(sigma)

    exparg = - 0.5 * np.einsum("j...,...j", delta, np.einsum("...jk,k...",  sigma_inv, delta))
    
    Norm = 1/np.sqrt(np.linalg.det(sigma*2*np.pi))
    if MP:
       
        G_mp = np.zeros(exparg.shape,dtype='complex')
        for i in range(exparg.shape[0]):
            for j in range(exparg.shape[1]):
                G_mp[i,j] = complex(mp.fprod([Norm, mp.exp(exparg[i,j])]))
        return G_mp
    else: 
        return Norm * np.exp(exparg)
def get_wigner_coherent_comb(data, x, p, MP = False):
    means, cov, weights  = data
    W = 0
        
    for i, mu in enumerate(means):
        W += weights[i] * Gauss(cov, mu, x, p, MP)
    return W


def plot_wigner_coherent_comb(ax, data, xvec, pvec, colorbar = True, GKP = False):
    
    #W = mp.chop(get_wigner_coherent_comb(data, xvec, pvec))
    W = get_wigner_coherent_comb(data,xvec,pvec)
    
    #if MP:
        #W = np.array([mp.re(j) for i in W for j in i]).reshape(len(xvec),len(pvec)) 
        #scale = np.max(W)
        
    #else:
        
    scale = np.max(W.real)
                                                            

    nrm = mpl.colors.Normalize(-scale, scale)

    if GKP:
        ax.contourf(xvec /np.sqrt(hbar * np.pi), pvec /np.sqrt(hbar * np.pi), W, 100, cmap=cm.RdBu, norm = nrm)
        ax.set_xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
        ax.set_ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
        ax.grid('on')
    else:
    
        ax.contourf(xvec, pvec, W, 100, cmap=cm.RdBu, norm = nrm)
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$p$", fontsize=12)
    if colorbar:
        plt.colorbar(cm.ScalarMappable(norm = nrm, cmap = cm.RdBu), ax = ax, shrink = 0.82)
    
    ax.set_aspect("equal")


def plot_marginal(ax, W, x, p, title, which = 'x', GKP = 'rect', ls='solid', lw=1,lab=None):
   
    marginal_x = np.sum(W,axis=0)*np.diff(p)[0]
    marginal_p = np.sum(W,axis=1)*np.diff(x)[0]

    W = np.round(W.real, 4)
    scale = np.max(W.real)
    nrm = mpl.colors.Normalize(-scale, scale)


    if GKP == 'rect':
        grid = np.sqrt(hbar*np.pi)
        if which == 'x':
        
            ax.plot(x/grid, marginal_x, linewidth=lw, linestyle=ls,label=lab)
            ax.set_xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
            ax.grid('on')
            ax.set_ylabel(r'$P(x)$')
    
        elif which == 'p':
            ax.plot(p/grid, marginal_p,linewidth=lw, linestyle=ls,label=lab)
            ax.set_xlabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
            ax.grid('on')
            ax.set_ylabel(r'$P(p)$')
            
    elif GKP =='square':
        grid = np.sqrt(hbar*np.pi/2)
        if which == 'x':
        
            ax.plot(x/grid, marginal_x,linewidth=lw, linestyle=ls,label=lab)
            ax.set_xlabel(r"$x(\sqrt{\hbar\pi/2})^{-1}$", fontsize=12)
            ax.grid('on')
            ax.set_ylabel(r'$P(x)$')
    
        elif which == 'p':
            ax.plot(x/grid, marginal_p,linewidth=lw, linestyle=ls,label=lab)
            ax.set_xlabel(r"$p(\sqrt{\hbar\pi/2})^{-1}$", fontsize=12)
            ax.grid('on')
            ax.set_ylabel(r'$P(p)$')
    ax.set_title(title,fontsize=12)
            
    return 


def plot_individual_gauss(W):
    fig, axes  = plt.subplots(1,2,figsize=(3,3),sharex=True,sharey=True)
    scale1 = np.max(W.real)
    scale2 = np.max(W.imag)
                                                            

    nrm1 = mpl.colors.Normalize(-scale1, scale1)
    nrm2 = mpl.colors.Normalize(-scale2, scale2)
    
    axes[0].contourf(x,x, W.real, 100, cmap=cm.RdBu, norm = nrm1)
    axes[1].contourf(x,x, W.imag, 100, cmap=cm.RdBu, norm = nrm2)
    axes[0].set_aspect("equal")
    axes[1].set_aspect("equal")
    fig.set_tight_layout(True)
    
    plt.show()

def make_sampling_plot(vals, reject_vals, state, 
                       axis = 0, norm = 1, method = 'normal', M =0 , covmat = [], prec =False):

    modes = [0]
    means_quad, covs_quad, quad_ind = select_quads(state, modes, covmat)

    if method != 'gaussian':
        ub_ind, ub_weights, ub_weights_prob = get_upbnd_weights(means_quad,
                                                                covs_quad, state.log_weights,method)
    else:
        cov_ub, mean_ub, scale = get_upbnd_gaussian(state, means_quad, covs_quad, quad_ind, prec)
        prefactor = 1/np.sqrt(2*np.pi*np.linalg.det(cov_ub))
        if M == 0:
            ub_weight = scale/prefactor
        else:
            ub_weight = M
        
        
    no_samples = len(vals)
    fig, axes = plt.subplots(1,1, figsize = (5,4))

    v = vals
    x= np.linspace(np.min(v), np.max(v),500)
        
    dx = np.diff(x)[-1]
    
    prob = np.zeros(len(x))
    ub_prob = np.zeros(len(x))
    for i, xval in enumerate(x):
        prob[i] = generaldyne_probability(xval, means_quad, covs_quad, state.log_weights, prec).real[0]
        
        if method != 'gaussian':
            
            ub_prob[i] = generaldyne_probability(xval, means_quad[ub_ind,:].real, covs_quad, ub_weights, prec).real[0]
        else:
            ub_prob[i] = generaldyne_probability(xval, mean_ub, cov_ub, ub_weight).real

        
    axes.plot(x, prob/state.norm, 'k--',label = 'p(m)')
    print('sum p(m)', np.sum(prob/state.norm)*np.diff(x)[0])
    
    axes.plot(x, ub_prob/state.norm, 'r',linestyle=':', label = 'upper bound dist')
    print('sum prob_upbnd', np.sum(ub_prob/state.norm)*np.diff(x)[0])


    
    if np.shape(vals[-1]) == (2,):
        shots = len(vals[:,0])
        nbins = int(shots/10)
        n,b,p = axes.hist(vals[:,0],bins = nbins, density =True, alpha = 0.9, label = 'accepted samples')
        n,b,p = axes.hist(reject_vals[:,0],bins = nbins, 
                             density =True, alpha = 0.5, label='rejected samples')
    else:
        shots = len(vals)
        nbins = int(shots/10)
        n,b,p = axes.hist(vals, bins = nbins, density =True, alpha = 0.9, label = 'accepted samples')
        n,b,p = axes.hist(reject_vals,bins = nbins,
                             density =True, alpha = 0.5, label='rejected samples')
        
    
    axes.set_xlabel('m')
    axes.set_ylabel('P(m)')
    axes.grid('on')
    plt.legend()
    #axes.set_ylim([0,1])
    #axes.set_xlim([np.min(v),np.max(v)])
    #axes.set_xlim([-1,1])

    
    