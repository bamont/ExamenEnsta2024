from PIL import Image 
import numpy as np
from numpy import linalg
import scipy as sp
from scipy import sparse
import time

gray_img = "example.bmp"
marked_img = "example_marked.bmp"
output = "example.png"
gray_img = "vieux.bmp"
marked_img = "vieux_m.bmp"
output = "vieux.png"

niters = 50_000
epsilon = 1.E-10

def compute_means(intensity):
    """
    Calcul la moyenne de l'intensite d'un pixel et de ses voisins immediats (diagonale comprise)
    """
    means = np.empty(intensity.shape,dtype=np.double)
    # Traitement des points internes (qui ne sont pas au bord de l'image)
    means[1:-1,1:-1] = np.array([[(1./9.)*np.sum(intensity[i-1:i+2,j-1:j+2])
            for j in range(1,intensity.shape[1]-1)] for i in range(1,intensity.shape[0]-1)])
    # Traitement des bords (mais qui ne sont pas dans un coin de l'image)
    means[0,1:-1] = np.array([(1./6.)*np.sum(intensity[0:2,j-1:j+2]) for j in range(1,intensity.shape[1]-1)])
    means[-1,1:-1] = np.array([(1./6.)*np.sum(intensity[-2:,j-1:j+2])
                                             for j in range(1,intensity.shape[1]-1)])
    means[1:-1,0] = np.array([(1./6.)*np.sum(intensity[i-1:i+2,0:2])
                                             for i in range(1,intensity.shape[0]-1)])
    means[1:-1,-1] = np.array([(1./6.)*np.sum(intensity[i-1:i+2,-2:])
                                             for i in range(1,intensity.shape[0]-1)])
    # Traitement des quatre pixels aux coins de l'image
    means[0,0] = 0.25*(intensity[0,0]+intensity[0,1]+intensity[1,0]+intensity[1,1])
    means[-1,0]= 0.25*(intensity[-1,0]+intensity[-1,1]+intensity[-2,0]+intensity[-2,1])
    means[0,-1] = 0.25*(intensity[0,-1]+intensity[0,-2]+intensity[1,-1]+intensity[1,-2])
    means[-1,-1]= 0.25*(intensity[-1,-1]+intensity[-1,-2]+intensity[-2,-1]+intensity[-2,-2])
    return means


def compute_variance(intensity, means):
    """
    Calcul la variance de l'intensite pour chaque pixel et de ses voisins immediats
    """
    variance = np.empty(intensity.shape,dtype=np.double)
    # Calcul variance pixels internes a l'image
    variance[1:-1,1:-1] = np.array([[np.sum(np.power(intensity[i-1:i+2,j-1:j+2]-means[i,j],2))
            for j in range(1,intensity.shape[1]-1)] for i in range(1,intensity.shape[0]-1)])
    # Calcul variance pixels aux bords de l'image
    variance[0,1:-1] = np.array([np.sum(np.power(intensity[0:2,j-1:j+2]-means[0,j],2)) for j in range(1,intensity.shape[1]-1)])
    variance[-1,1:-1] = np.array([np.sum(np.power(intensity[-2:,j-1:j+2]-means[-1,j],2)) for j in range(1,intensity.shape[1]-1)])
    variance[1:-1,0] = np.array([np.sum(np.power(intensity[i-1:i+2,0:2]-means[i,0],2)) for i in range(1,intensity.shape[0]-1)])
    variance[1:-1,-1] = np.array([np.sum(np.power(intensity[i-1:i+2,-2:]-means[i,-1],2)) for i in range(1,intensity.shape[0]-1)])
    # Calcul variance pour les quatre pixels aux coins de l'image
    variance[0,0] = ((intensity[0,0]-means[0,0])**2+(intensity[0,1]-means[0,0])**2+(intensity[1,0]-means[0,0])**2+(intensity[1,1]-means[0,0])**2)
    variance[-1,0]= ((intensity[-1,0]-means[-1,0])**2+(intensity[-1,1]-means[-1,0])**2+(intensity[-2,0]-means[-1,0])**2+(intensity[-2,1]-means[-1,0])**2)
    variance[0,-1] = ((intensity[0,-1]-means[0,-1])**2+(intensity[0,-2]-means[0,-1])**2+(intensity[1,-1]-means[0,-1])**2+(intensity[1,-2]-means[0,-1])**2)
    variance[-1,-1]= ((intensity[-1,-1]-means[-1,-1])**2+(intensity[-1,-2]-means[-1,-1])**2+(intensity[-2,-1]-means[-1,-1])**2+(intensity[-2,-2]-means[-1,-1])**2)
    return variance


def compute_wrs(intensity, means, variance, ir, jr, ic, jc):
    """
    Calcul un poids pour la contribution d'un pixel voisin (ic,jc) au pixel de coordonne (ir,jr)
    en fonction de la correlation de l'intensite du pixel (ic,jc) avec le pixel (ir,jr)
    """
    nx = intensity.shape[1]
    index_r = ir*nx + jr
    index_c = ic*nx + jc
    # Prise en compte de la variance quand elle est nulle
    sigma = max(variance[ir,jr],0.000002)
    mu_r  = means[ir,jr]
    return 1.+(intensity[ir,jr]-mu_r)*(intensity[ic,jc]-mu_r)/sigma

def assembly_row( intensity, means, variance, pos, pt_rows, ind_cols, coefs):
    """
    Assemble la ligne de la matrice correspondant au pixel se trouvant a la position pos = (i,j)

    Le stockage de la matrice est un stockage morse, c'est à dire :

    pt_rows represente le debut de chaque ligne de la matrice dans les tableaux ind_cols et coefs
    ind_cols represente les indices colonnes de chaque element non nul de la matrice
    coefs stocke les coefficients non nuls de la matrice
    """
    nx = intensity.shape[1]
    ny = intensity.shape[0]
    # Calcul nombre coefficients non nuls sur la ligne de la matrice :
    nnz = 9
    if ( pos[0] == 0 or pos[0] == ny-1 ) and ( pos[1] == 0 or pos[1] == nx-1):
        nnz = 4 
    elif ( pos[0] == 0 or pos[0] == ny-1 ) and not ( pos[1] == 0 or pos[1] == nx-1):
        nnz = 6
    elif not ( pos[0] == 0 or pos[0] == ny-1 ) and ( pos[1] == 0 or pos[1] == nx-1):
        nnz = 6
    index = pos[0]*nx+pos[1]
    # Calcul de la position de la ligne suivante dans la matrice :
    pt_rows[index+1] = pt_rows[index] + nnz
    # On commence a remplir ind_cols et coefs avec les coefficinets adequats pour la matrice
    start = pt_rows[index]
    sum = 0.
    if pos[0] >0:
        if pos[1] > 0:
            ind_cols[start] = index - nx - 1
            wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]-1,pos[1]-1)
            sum += wrs
            coefs[start] = -wrs
            start += 1
        ind_cols[start] = index - nx
        coefs[start] = -compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]-1,pos[1])
        start += 1
        if pos[1] < nx-1:
            ind_cols[start] = index - nx + 1
            coefs[start] = -compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]-1,pos[1]+1)
            start += 1
    if pos[1] > 0:
        ind_cols[start] = index - 1
        coefs[start] = -compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0],pos[1]-1)
        start += 1
    ind_cols[start] = index
    coefs[start] = +1.
    start += 1
    if pos[1] < nx-1:
        ind_cols[start] = index + 1
        coefs[start] = -compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0],pos[1]+1)
        start += 1
    if pos[0] < ny-1:
        if pos[1] > 0:
            ind_cols[start] = index + nx - 1
            coefs[start] = -compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]+1,pos[1]-1)
            start += 1
        ind_cols[start] = index + nx
        coefs[start] = -compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]+1,pos[1])
        start += 1
        if pos[1] < nx-1:
            ind_cols[start] = index + nx + 1
            coefs[start] = -compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]+1,pos[1]+1)
            start += 1


def compute_matrix(intensity, means, variance):
    """
    Calcul la matrice issue de la minimisation de la fonction quadratique
    """
    ny = intensity.shape[0]
    nx = intensity.shape[1]
    # Dimension de la matrice
    dim  = intensity.shape[0] * intensity.shape[1]
    # Nombre d'elements non nuls prevus pour la matrice :
    nnz = 9*(nx-2)*(ny-2) + 12*((nx-2)+(ny-2)) + 16
    # Indices du début des lignes dans les tableaux indCols et coefficients :
    beg_rows = np.zeros(dim+1, dtype=np.int64)
    # Indices colonnes des elements non nuls :
    ind_cols = np.empty(nnz, dtype=np.int64)
    coefs    = np.empty(nnz, dtype=np.double)

    # Pour chaque pixel (irow, icol); on fait correspondre la ligne de la matrice d'indice irow*nx + icol
    # On assemble la matrice ligne par ligne
    for irow in range(ny):
        for jcol in range(nx):
            assembly_row(intensity, means, variance, (irow,jcol), beg_rows, ind_cols, coefs)
    assert(beg_rows[-1] == nnz)
    # On normalise les poids wrs hors diagonale ligne par ligne:
    for irow in range(nx*ny):
        sum = 0.
        for ptrow in range(beg_rows[irow],beg_rows[irow+1]):
            sum += np.abs(coefs[ptrow]) if ind_cols[ptrow] != irow else 0.
        for ptrow in range(beg_rows[irow],beg_rows[irow+1]):
            if ind_cols[ptrow] != irow:
                coefs[ptrow] /= sum
    # On retourne la matrice sous forme d'une matrice creuse stockee en csr avec scipy
    return sparse.csr_matrix((coefs, ind_cols, beg_rows), dtype=np.double)

def search_fixed_colored_pixels(image_marquee):
    """
    Recherche dans l'image marquee l'indice des pixels dont on a fixé la couleur:
    On utilise pour cela l'espace colorimetrique HSV qui separe bien l'intensite
    de la saturation et de la teinte pour chaque pixel :
    """
    global marked_img
    im = Image.open(marked_img)
    im = im.convert('HSV')
    values = np.array(im)
    hue        = (1/255)*np.array(values[:,:,0].flat, dtype=np.double)
    saturation = (1/255)*np.array(values[:,:,1].flat, dtype=np.double)
    return np.nonzero((hue != 0.) * (saturation != 0.))[0]

def apply_dirichlet(A : sparse.csr_matrix, dirichlet : np.array):
    """
    Applique une condition de dirichlet aux endroits ou la couleur est deja definie a l'initiation
    """
    for irow in range(A.shape[0]):
        if irow in dirichlet:
            A.data[A.indptr[irow]:A.indptr[irow+1]] = [0. if A.indices[i]!=irow else 1. for i in range(A.indptr[irow],A.indptr[irow+1])]
        else:
            for jcol in range(A.indptr[irow],A.indptr[irow+1]):
                if A.indices[jcol] in dirichlet:
                    A.data[jcol] = 0.


def minimize( A : sparse.csr_matrix, b : np.array, x0 : np.array, niters : int, epsilon : float):
    """
    Minimise la fonction quadratique a l'aide d'un gradient conjugue
    """
    r = b-A.dot(x0)
    nrm_r0 = linalg.norm(r)
    gc = A.transpose().dot(r)
    x = np.copy(x0)
    p = np.copy(gc)
    cp = A.dot(p)
    nrm_gc = linalg.norm(gc)
    nrm_cp = linalg.norm(cp)
    alpha = nrm_gc*nrm_gc/(nrm_cp*nrm_cp)
    x += alpha*p
    r -= alpha*cp
    nrm_r = linalg.norm(r)
    gp = np.copy(gc)
    nrm_gp = nrm_gc
    gc = A.transpose().dot(r)
    for i in range(1,niters):
        print(f"Iteration {i:06}/{niters:06} -> ||r||/||r0|| = {nrm_r/nrm_r0:16.14}",end='\r')
        nrm_gc = linalg.norm(gc)
        if nrm_gc < 1.E-14: return x
        beta = -nrm_gc*nrm_gc/(nrm_gp*nrm_gp)
        p = gc - beta*p
        cp = A.dot(p)
        nrm_cp = linalg.norm(cp)
        alpha = nrm_gc*nrm_gc/(nrm_cp*nrm_cp)
        x += alpha*p
        r -= alpha*cp
        gp = np.copy(gc)
        nrm_gp = nrm_gc
        gc = A.transpose().dot(r)
        nrm_r = linalg.norm(r)
        if nrm_r < epsilon*nrm_r0: break 
    return x

# On va lire l'intensite des pixels dans l'image en teinte de gris :
im = Image.open(gray_img)
im = im.convert('HSV')

values = np.array(im)
# Intensite qu'on normalise avec des valeurs entre 0 et 1 :
intensity = (1/255)*np.array(values[:,:,2],dtype=np.double)

# On va maintenant extraire les pixels colorises dans l'image marquee :
im = Image.open(marked_img)
im = im.convert('YCbCr')
values = np.array(im)
# Les composantes Cb (bleu) et Cr (Rouge) sont normalisees :
Cb = (1/255)*np.array(values[:,:,1].flat, dtype=np.double)
Cr = (1/255)*np.array(values[:,:,2].flat, dtype=np.double)

# Calcul de l'intensite moyenne d'un pixel avec ses voisins immediats :
deb = time.time()
means = compute_means(intensity)
tps_mean = time.time() - deb
print(f"Temps calcul fonction means : {tps_mean}")

# Calcul de la variance de l'intensite d'un pixel avec ses voisins immediats :
deb = time.time()
sigma = compute_variance(intensity, means)
tps_sigma = time.time() - deb
print(f"Temps calcul variance : {tps_sigma}")

# Calcul de la matrice issue de la minimisation de la forme quadratique J(U) sans les conditions de dirichlet:
deb=time.time()
A = compute_matrix(intensity, means, sigma)
tps_assemblage = time.time() - deb
print(f"Temps pris pour assemblage matrice : {tps_assemblage}")

# On cherche les pixels dont on a fixe la couleur => ils correspondent a une condition de dirichlet :
dirichlet = search_fixed_colored_pixels(marked_img)
# Calcul seconds membres :
print("Prise en compte des conditions de Dirichlet")
# Applique dirichlet à la matrice :
deb=time.time()
b_Cb = -A.dot(Cb)
b_Cr = -A.dot(Cr)
apply_dirichlet(A, dirichlet)
tps_dirichlet = time.time()-deb
print(f"Temps pris pour application dirichlet sur matrice : {tps_dirichlet}")

print(f"Minimisation de la quadratique pour la composante Cb de l'image couleur")
deb=time.time()
x0 = np.zeros(Cb.shape,dtype=np.double)
new_Cb = Cb + minimize(A, b_Cb, x0, niters,epsilon)
print(f"\nTemps calcul min Cb : {time.time()-deb}")

print(f"Minimisation de la quadratique pour la composante Cr de l'image couleur")
deb=time.time()
x0 = np.zeros(Cr.shape,dtype=np.double)
new_Cr = Cr + minimize(A, b_Cr, x0, niters,epsilon)
print(f"\nTemps calcul min Cr : {time.time()-deb}")

# On remet les valeurs des trois composantes de l'image couleur YCbCr entre 0 et 255 :
new_Cb *= 255.
new_Cr *= 255.
intensity *= 255.

# Puis on sauve l'image dans un fichier :
new_image_array = np.empty((intensity.shape[0],intensity.shape[1],3), dtype=np.uint8)
new_image_array[:,:,0] = intensity.astype('uint8')
new_image_array[:,:,1] = np.reshape(new_Cb, intensity.shape).astype('uint8')
new_image_array[:,:,2] = np.reshape(new_Cr, intensity.shape).astype('uint8')
new_im = Image.fromarray(new_image_array, mode='YCbCr')
new_im.convert('RGB').save(output, 'PNG')
