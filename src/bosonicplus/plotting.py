import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from mpmath import mp, fp
hbar = 2

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

def plot_wig(ax, W, q1, q2, colorbar = True, xlabel = None, ylabel = None, GKP = False):
    
    W = np.round(W.real, 4)
    scale = np.max(W.real)
    nrm = mpl.colors.Normalize(-scale, scale)
    if GKP:
        im = ax.contourf(q1 /np.sqrt(hbar * np.pi), q2 /np.sqrt(hbar * np.pi), W, 100, cmap=cm.RdBu, norm = nrm)
        ax.set_xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
        ax.set_ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
        ax.grid('on')
    else:
        im = ax.contourf(q1, q2, W, 100, cmap=cm.RdBu, norm = nrm)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_xlabel(r'$x$', fontsize=12)
            ax.set_ylabel(r'$p$', fontsize=12)
    
    #ax.set_xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
    #ax.set_ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
    
    if colorbar:
        plt.colorbar(cm.ScalarMappable(norm = nrm, cmap = cm.RdBu), ax = ax, shrink = 0.82)
    
    ax.set_aspect("equal")
    return im


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

def plot_wigner_marginals(W, x, p, title, GKP='rect'):
    ## To do: add colorbar
    # Start with a square Figure.
    fig = plt.figure(figsize=(6,6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal Axes and the main Axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2,2,  width_ratios=(1,4), height_ratios=(1,4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[0, 1], sharex = ax)
    ax_p = fig.add_subplot(gs[1, 0], sharey = ax)
    
    marginal_x = np.sum(W,axis=0)*np.diff(p)[0]
    marginal_y = np.sum(W,axis=1)*np.diff(x)[0]

    W = np.round(W.real, 4)
    scale = np.max(W.real)
    nrm = mpl.colors.Normalize(-scale, scale)

    if GKP == 'rect':
        grid = np.sqrt(hbar*np.pi)
        ax.contourf(x /grid, p/grid, W, 100, cmap=cm.RdBu, norm = nrm)
        ax.set_xlabel(r"$x(\sqrt{\hbar\pi})^{-1}$", fontsize=15)
        #ax.set_ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=12)
        ax_p.set_ylabel(r"$p(\sqrt{\hbar\pi})^{-1}$", fontsize=15)
        ax.grid('on')
    elif GKP =='square':
        grid = np.sqrt(hbar*np.pi/2)
        ax.contourf(x/grid, p/grid, W, 100, cmap=cm.RdBu, norm = nrm)
        ax.set_xlabel(r"$x(\sqrt{\hbar\pi/2})^{-1}$", fontsize=15)
        #ax.set_ylabel(r"$p(\sqrt{\hbar\pi/2})^{-1}$", fontsize=12)
        ax_p.set_ylabel(r"$p(\sqrt{\hbar\pi/2})^{-1}$", fontsize=15)
        ax.grid('on')
    else:
        ax.contourf(x, p, W, 100, cmap=cm.RdBu, norm = nrm)
        grid = 1
        ax.set_xlabel(r"$x$", fontsize=15)
        ax_p.set_ylabel(r"$p$", fontsize=15)

    ax.set_aspect("equal")

    
    ax_x.plot(x/grid, marginal_x)
    ax_p.plot(marginal_y, x/grid)
    
    ax_x.tick_params(axis = 'x',labelbottom = False)
    ax.tick_params(axis = 'y', labelleft=False)
    ax_x.grid('on')
    ax_p.grid('on')
    ax_x.set_ylabel(r'$P(x)$')
    ax_p.set_xlabel(r'$P(p)$')
    ax_p.invert_xaxis()
    plt.suptitle(title)
    fig.tight_layout()
    
    return fig, ax, ax_x, ax_p


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

    
    