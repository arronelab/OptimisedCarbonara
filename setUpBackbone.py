import biobox as bb
import torch
import plotly.graph_objects as go
from PSIPREDauto.functions import single_submit
import os, shutil
import numpy as np
import rmsd


def pdb_to_ca(pdb_fl,chain='A'):
    """ Reads in a PDB file, returns the CA backbone for a given chain 

    Args:
        pdb_fl (str): PDB file path.
        chain (str): Chain ID of interest.

    Returns: 
        (torch.tensor), size ([N,3]).
    """
    mol = bb.Molecule(pdb_fl)
    idx1 = (mol.data['name']=='CA').values
    idx2 = (mol.data['name']=='CA A').values
    CA = torch.from_numpy(mol.coordinates[0][np.logical_or(idx1,idx2)].copy())
    return CA

def get_neighbour_distances(CA):
    """ Computes the distance between residue i and residue i+1.

    Args:
        CA (torch.tensor): CA backbone, size ([N,3])

    Returns: 
        (torch.tensor), size ([N-1]).
    """
    cdist = torch.cdist(CA,CA)
    return cdist[1:,].diagonal()

def get_nonneighbour_distances(CA):
    """ Computes the distances between residues i and j, where j>i+1. 
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([(N-2)*(N-1)/2])
    """
    cdist = torch.cdist(CA,CA)
    N = cdist.shape[0]
    mask = np.triu(np.ones((N, N), dtype=bool), k=2)  # Mask for j > i + 1
    nonneighbs = cdist[mask]  # Apply the mask to the tensor
    return nonneighbs

def curvature(CA):
    """ Computes the curvature of the subsection (i,i+1,i+2,i+3)
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([N-3])
    """
    v1 = CA[:-3]
    v2 = CA[1:-2]
    v3 = CA[2:-1]
    v4 = CA[3:]
    m1 = (v2+v1)/2
    m2 = (v3+v2)/2
    m3 = (v4+v3)/2
    v1 = m1-m3
    v2 = m2-m3
    cross = torch.cross(v1,v2,dim=-1)
    sin_theta = torch.linalg.norm(cross,axis=-1)/(torch.linalg.norm(v1,axis=-1)*torch.linalg.norm(v2,axis=-1))
    return (2*torch.absolute(sin_theta))/torch.linalg.norm(m1-m2,axis=-1)

def torsion(v):
    """ Computes the torsion of the subsection (i,i+1,i+2,i+3)
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([N-3])
    """
    v1 = v[:-3]
    v2 = v[1:-2]
    v3 = v[2:-1]
    v4 = v[3:] 
    e1 = v2-v1
    e2 = v3-v2
    e3 = v4-v3
    n1 = torch.nn.functional.normalize(torch.cross(e1,e2,dim=-1),dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(e2,e3,dim=-1),dim=-1)
    cos_theta = (n1*n2).sum(dim = -1)
    theta = torch.arccos(cos_theta)
    idx = torch.where((torch.cross(n1,n2,dim=-1)*e2).sum(dim = -1)<0)
    theta[idx] = theta[idx]*-1
    length = (torch.linalg.norm(e1,axis=-1)+torch.linalg.norm(e2,axis=-1)+torch.linalg.norm(e3,axis=-1))/3
    return (2/length)*torch.sin(theta/2)

def wij_matrix(CA):
    """ Computes the the exact evalutation of the Gauss integral over line segments i - i+1, j - j+1
    
    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
        
    Returns: 
        (torch.tensor), size ([N-1,N-1])
    """
    starts = CA[:-1]
    ends = CA[1:]
    A = starts.reshape(1,-1,3) - starts.reshape(-1,1,3)
    B = ends.reshape(1,-1,3) - ends.reshape(-1,1,3)
    C = ends.reshape(1,-1,3) - starts.reshape(-1,1,3)
    n1 = torch.nn.functional.normalize(torch.cross(A,C,dim=-1),dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(C,B,dim=-1),dim=-1)
    n3 = torch.nn.functional.normalize(torch.cross(B,-C.transpose(1,0),dim=-1),dim=-1)
    n4 = torch.nn.functional.normalize(torch.cross(-C.transpose(1,0),A,dim=-1),dim=-1)
    arg1 = (n1*n2).sum(dim = -1)
    arg2 = (n2*n3).sum(dim = -1)
    arg3 = (n3*n4).sum(dim = -1)
    arg4= (n4*n1).sum(dim = -1)
    eps = 1e-7
    arg1 = torch.clamp(arg1,-1+eps,1-eps)
    arg2 = torch.clamp(arg2,-1+eps,1-eps)
    arg3 = torch.clamp(arg3,-1+eps,1-eps)
    arg4 = torch.clamp(arg4,-1+eps,1-eps)
    omega_star = torch.arcsin(arg1) + torch.arcsin(arg2) + torch.arcsin(arg3) + torch.arcsin(arg4)
    Cii = ends-starts
    r12 = Cii.reshape(-1,1,3)
    r34 = Cii.reshape(1,-1,3)
    sign_arg = (torch.cross(r34,r12,dim=-1)*A).sum(dim=-1)
    return (omega_star*torch.tanh(sign_arg))/(4*torch.pi)

def secondary_structure_transform_matrix(x,y,gamma):
    """ Computes the matrix for a rigid transorm of e1 to a given subsection.

    Args:
        x (torch.tensor): First point of subsection, size ([1,3])
        y (torch.tensor): Last point of subsection, size ([1,3])
        gamma (torch.tensor): Angle of rotation about the axis y-x, size ([1])
    
    Returns: 
        (torch.tensor), size ([3,3])
    """
    t = y-x
    r = torch.functional.norm(t)
    theta = torch.arccos(t[2]/r)
    phi = torch.sign(t[1])*torch.arccos(t[0]/torch.functional.norm(t[:2]))
    alpha = phi
    beta = -(torch.pi/2-theta)
    roll = torch.tensor([[1.,0.,0.],
                         [0.,torch.cos(gamma),-torch.sin(gamma)],
                        [0., torch.sin(gamma),torch.cos(gamma)]])
    pitch = torch.tensor([[torch.cos(beta),0.,torch.sin(beta)],
                         [0.,1.,0.],
                          [-torch.sin(beta),0.,torch.cos(beta)]])
    yaw = torch.tensor([[torch.cos(alpha),-torch.sin(alpha),0.],
                       [torch.sin(alpha),torch.cos(alpha),0.],
                       [0.,0.,1.]])
    scale = torch.tensor([[r,0,0],[0,1,0],[0,0,1]])
    return yaw@pitch@roll@scale

def neighbouring_distances_gaussian(CA,t):
    """ Evaluates the log of a Normal PDF for each neighbouring distance of the CA backbone.

    Args:
        CA (torch.tensor): CA backbone, size ([N,3])
    
    Returns:
        (torch.tensor), size([N-1])
    """
    mean = torch.tensor(3.80523548)
    std = torch.tensor(0.02116009)+t
    neighb_norm = torch.distributions.Normal(mean,std)
    return neighb_norm.log_prob(get_neighbour_distances(CA))

def nonneighbouring_distances_gaussian(CA,t):
    """ Evaluates the log of a Logistic CDF for each nonneighbouring distance of the CA backbone.

    Args:
        CA (torch.tensor): CA backbone, size([N,3])

    Returns: 
        (torch.tensor), size([(N-2)*(N-1)/2])
    """
    mean = torch.tensor(3.97512675)
    std = torch.tensor(0.12845008)+t
    base_distribution = torch.distributions.Uniform(0, 1)
    transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=mean, scale=std)]
    logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
    return torch.log(logistic.cdf(get_nonneighbour_distances(CA)))
    
def curvature_torsion_gmm(CA,t):
    """ Evaluates the log of a GaussianMixture CDF for each (curvature, torsion) pair along the CA backbone.

    Args:
        CA (torch.tensor): CA backbone, size([N,3])

    Returns: 
        (torch.tensor), size ([N-3])
    """
    means = [[ 0.04687773, -0.52137024],
       [ 0.46548875,  0.22172971],
       [ 0.1277566 ,  0.49774274],
       [ 0.23812849, -0.38673841],
       [ 0.41870451,  0.12128396],
       [ 0.32275806,  0.33953025],
       [ 0.16693615, -0.45908017],
       [ 0.31653644, -0.01269496],
       [ 0.24136913,  0.42854014],
       [ 0.04827249,  0.52317198],
       [ 0.28830156,  0.23408932],
       [ 0.45444073,  0.20289183],
       [ 0.09652552, -0.50411539],
       [ 0.4424514 ,  0.26991139],
       [ 0.25414876, -0.29170993]]
    
    gmm_means = torch.tensor(means)
    covs=[[[ 3.85547718e-04,  6.18552928e-05],
        [ 6.18552928e-05,  2.23668495e-05]],

       [[ 1.04380877e-04, -3.84656455e-05],
        [-3.84656455e-05,  2.23444482e-04]],

       [[ 1.66693748e-03, -5.75160089e-04],
        [-5.75160089e-04,  2.99601176e-04]],

       [[ 1.58991438e-03,  1.04387662e-04],
        [ 1.04387662e-04,  1.96502993e-03]],

       [[ 1.04979619e-03,  3.42509622e-04],
        [ 3.42509622e-04,  3.59133854e-03]],

       [[ 3.09090465e-03, -1.18157019e-03],
        [-1.18157019e-03,  1.82729529e-03]],

       [[ 1.33971624e-03,  5.34469942e-04],
        [ 5.34469942e-04,  9.31168134e-04]],

       [[ 5.77141632e-03,  1.43790956e-03],
        [ 1.43790956e-03,  9.17204485e-03]],

       [[ 2.29005159e-03, -1.45952513e-03],
        [-1.45952513e-03,  1.27571080e-03]],

       [[ 5.89647517e-04, -5.08594386e-05],
        [-5.08594386e-05,  1.77335427e-05]],

       [[ 1.15772745e-02, -8.80155561e-03],
        [-8.80155561e-03,  1.78539875e-02]],

       [[ 2.38313560e-04, -5.52107751e-06],
        [-5.52107751e-06,  5.74140280e-04]],

       [[ 5.77095483e-04,  1.92114463e-04],
        [ 1.92114463e-04,  1.78576718e-04]],

       [[ 6.37962181e-04,  5.56429157e-05],
        [ 5.56429157e-05,  1.09551888e-03]],

       [[ 1.04518003e-02,  4.95213083e-03],
        [ 4.95213083e-03,  9.25080667e-03]]]
    gmm_covs = torch.tensor(covs)+t
    mix = torch.distributions.Categorical(torch.tensor([0.10551554, 0.360049  , 0.02621649, 0.03810681, 0.03434352,
       0.02114176, 0.06677582, 0.02202394, 0.03291277, 0.04369073,
       0.031766  , 0.08337998, 0.06779723, 0.04577114, 0.02050928]))
    comp = torch.distributions.MultivariateNormal(gmm_means, gmm_covs)
    gmm = torch.distributions.MixtureSameFamily(mix, comp)
    kap = curvature(CA)
    tau = torsion(CA)
    kt = torch.stack((kap,tau),dim=-1)
    return gmm.log_prob(kt)

def build_SKMT_backbone(params,nonlinker_reps):
    """Given the linker parameters and gammas, and the canonical nonlinker representatives
    builds a CA backbone tensor."""
    linker_params = params[0]
    gammas = params[1]
    CA = []
    for i in range(len(linker_params)-1):
        if len(linker_params[i])>2:
            CA.append(linker_params[i][0].unsqueeze(0))
            CA.append(linker_params[i][int(len(linker_params[i])/2)].unsqueeze(0))
        else:
            CA.append(linker_params[i][0].unsqueeze(0))
        mat = secondary_structure_transform_matrix(linker_params[i][-1],linker_params[i+1][0],gammas[i]).T
        nonlink = linker_params[i][-1] + nonlinker_reps[i]@mat
        CA.append(nonlink[0].unsqueeze(0))
    CA.append(linker_params[-1][0].unsqueeze(0))
    if not torch.equal(CA[-1],linker_params[-1][-1]):
        CA.append(linker_params[-1][-1].unsqueeze(0))
    return torch.cat(CA,dim=0)

def acn_penalty(CA,t):
    """ Evaluates the log of a Normal PDF for the ACN of the CA backbone

    Args:
        CA (torch.tensor), size ([N,3])

    Returns:
        (tensor.float)
    """
    acn = torch.absolute(wij_matrix(CA)[torch.nonzero(wij_matrix(CA),as_tuple=True)]).sum()
    x = CA.shape[0]
    mean = torch.tensor((x/4)**1.4 - 6)
    std = torch.tensor(0.12444077*x+3.20058622)+t
    base_distribution = torch.distributions.Uniform(0, 1)
    transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=mean, scale=std)]
    logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
    return torch.log(logistic.cdf(acn))

def split_curve(CA,n=5):
    N = CA.shape[0]//n
    reshaped_CA = CA[:N*n].reshape(N,n,3)
    return reshaped_CA

def sub_acn(CA,n=5):
    split_CA = split_curve(CA,n)
    split_wij = torch.vmap(wij_matrix)(split_CA)
    n_acns = torch.absolute(split_wij).sum(dim=(1,2))
    shape = 0.9745809291648566  
    loc = -0.0041441164826397136 
    lognormal_dist = torch.distributions.LogNormal(loc, shape)
    return lognormal_dist.log_prob(n_acns)

def load_FP(fp_fl):
    """ Reads in a secondary structure fingerprint, with a simplified 3 letter code.
        H = Helix, S = Strand, - = Linker

    Args:
        fp_fl (str): fingeprint file location
    
    Returns:
        FP (str): secondary structure fingerprint 
    """
    FP = []
    with open(fp_fl) as file:
        for line in file:
            FP.extend(list(line.replace(" ", "")))
    FP = ''.join(FP)
    if FP[0]!='-':
        FP[0] = '-'
    if FP[-1]!='-':
        FP[-1] = '-'
    return FP

def get_linker_indices(FP):
    """ Finds the indices of linker residues. 

    Args:
        FP (str): secondary structure fingerprint

    Returns:
        (list)
    """
    return [i for i, c in enumerate(FP) if c=='-']

def find_intervals(nums):
    """ Computes the intervals corresponding to a list of integers containing gaps
        e.g. find_intervals([0,1,2,5,7,8,9]) = [[0,2],[5,5],[7,9]] 
    
    Args:
        nums (list): list of integers

    Returns:
        intervals (list of lists): list of (first integer, last integer) tuples
    """
    intervals = []
    start = None
    
    for i in range(len(nums)):
        if i == 0 or nums[i] != nums[i-1] + 1:
            if start is not None:
                intervals.append([start, nums[i-1]])
            start = nums[i]
    
    if start is not None:
        intervals.append([start, nums[-1]])
    
    return intervals

def get_linker_intervals(FP):
    """ Finds the start and end index of linker subsections.

    Args:
        FP (str): secondary structure fingerprint

    Returns:
        (list of lists): list of (start index, end index) tuples
    """
    linkers = get_linker_indices(FP)
    return find_intervals(linkers)

def get_nonlinker_intervals(FP):
    """ For each nonlinker subsection, finds the index of endpoint of the previous linker, and the index of the start point of the subsequent linker.

    Args
        FP (str): secondary structure fingerprint

    Returns:
        (list of lists): list of (start index, end index) tuples
    """
    lindex = get_linker_intervals(FP)
    return [[lindex[i][1],lindex[i+1][0]] for i in range(len(lindex)-1)]

def single_representative(nonlinker_subsec,x,y):
    """ Maps a given nonlinker subsection to a canonical representative lying between [0,1] on the x axis.

    Args:
        nonlinker_subsec (torch.tensor): Nonlinker subsection of CA backbone, size ([m,3])

    Returns:
        (torch.tensor), size([m,3])
    """
    return (nonlinker_subsec-x)@torch.linalg.inv(secondary_structure_transform_matrix(x,y,torch.tensor(0.,dtype=torch.double))).T

def nonlinker_representatives(CA,FP):
    """
    Given a CA backbone and secondary structure fingeprint, returns a list of the canonical representative
    of each nonlinker subsection.
    """
    nonlinker_intervals = get_nonlinker_intervals(FP)
    reps = []
    for i in nonlinker_intervals:
        reps.append(single_representative(CA[i[0]+1:i[1]],CA[i[0]],CA[i[1]]))
    return reps

def get_linker_params(CA,FP):
    """Returns a list of tensors, each tensor is the coordinates of linker subsection."""
    return [CA[i[0]:i[1]+1] for i in get_linker_intervals(FP)]

def get_params(CA,FP):
    """Returns a tuple of the linker subsections, and the gammas (initialised to 0)
    The gammas represent the rotation for the nonlinker representative."""
    linker_params = get_linker_params(CA,FP)
    gammas = [torch.tensor(0.,dtype=torch.double) for i in range(len(linker_params)-1)]
    return [linker_params,gammas]

def build_backbone(params,nonlinker_reps):
    """Given the linker parameters and gammas, and the canonical nonlinker representatives
    builds a CA backbone tensor."""
    linker_params = params[0]
    gammas = params[1]
    CA = []
    for i in range(len(linker_params)-1):
        CA.append(linker_params[i])
        mat = secondary_structure_transform_matrix(linker_params[i][-1],linker_params[i+1][0],gammas[i]).T
        nonlink = linker_params[i][-1] + nonlinker_reps[i]@mat
        CA.append(nonlink)
    CA.append(linker_params[-1])
    return torch.cat(CA,dim=0)

def plotMol(coords):
    """Plots a CA backbone tensor."""    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        marker=dict(
            size=1,
            color='black',
        ),
        line=dict(
            color='black',
            width=20
        )
    ))
    fig.update_layout(width=1000,height=1000)
    fig.update_layout(showlegend=False,
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            aspectratio = dict( x=1, y=1, z=1 ),
            aspectmode = 'manual',
            xaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),
            yaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),
            zaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),),
    )
    fig.show()

def pdb_to_fasta(pdb_file_loc,chain):
### Opens a PDB file using biobox, retrieves the primary sequence,
### writes it to a fasta file.
    aa3to1={
   'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
   'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
   'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
   'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
   'MSE':'M', 'HID': 'H', 'HIP': 'H'
    }
    M = bb.Molecule(pdb_file_loc)
    CA,idx  = M.atomselect(chain,'*','CA',get_index=True)
    aa = [aa3to1[i[0]] for i in M.get_data(idx,columns=['resname'])]
    with open(pdb_file_loc[:-4]+'.fasta','w+') as fout:
          fout.write(''.join(aa))
            
def get_secstruc_psipred(pdb_file_loc,chain):
### Using PSIPredauto, runs a PSIPred SecStruc prediction
### for a given PDB file.
    if not os.path.isfile(pdb_file_loc[:-4]+'.fasta'):
        pdb_to_fasta(pdb_file_loc,chain)
    if not os.path.isdir(pdb_file_loc[:-4]+'.fasta output/'):
        single_submit(pdb_file_loc[:-4]+'.fasta', "foo@bar.com", '')
        
def convert(s):
### Converts a list of letters into a word
    new = ""
    for x in s:
        new+= x
    return new

def simple_ss_clean(fp):
### Simple SecStruc clean up check for singleton SSEs.
    for i in range(len(fp)-1):
        if fp[i-1]==fp[i+1] and fp[i-1]!=fp[i]:
            fp[i]=fp[i-1]
    return convert(fp)

def get_ss_fp_psipred(fasta_file_loc):
### From the outpuit of the PSIPredrun run, converts to the
### simple 3 letter code SecStruc FP.
    dssp_to_simp = {"I" : "H",
                 "S" : "-",
                 "H" : "H",
                 "E" : "S",
                 "G" : "H",
                 "B" : "S",
                 "T" : "-",
                 "C" : "-"
                 }
    lines = []
    with open(fasta_file_loc+' output/'+os.path.splitext(os.path.basename(fasta_file_loc))[0]+'.ss','r') as fin:
        for line in fin:
            lines.append(line.split())
    ss = [dssp_to_simp[i[2]] for i in lines]
    return simple_ss_clean(ss)

def write_fingerprint_file(pdb_file_loc,chain):
### For a given PDB file, sets up a call to PSIPred to predict
### the SecStruc, then writes it to a file in the same dir as the PDB file.
    get_secstruc_psipred(pdb_file_loc,chain)
    ss = get_ss_fp_psipred(pdb_file_loc[:-4]+'.fasta')
    with open(os.path.dirname(pdb_file_loc)+'/fingerPrint1.dat','w+') as fout:
        fout.write(ss)
    os.remove(pdb_file_loc[:-4]+'.fasta')
    shutil.rmtree(pdb_file_loc[:-4]+'.fasta output')




def overlayMolSubsecs(start_coords,current_coords,subsec_start=0,subsect_end=-1):
    ### Plots a CA backbone tensor.
    start_coords = start_coords.numpy()
    current_coords = current_coords.numpy()
    #aligned = rmsd.kabsch_fit(current_coords,start_coords)
    diff_coords = np.array([np.linalg.norm(start_coords[i]-current_coords[i]) for i in range(len(start_coords))])
    subsec_start = max(0,diff_coords.argmax()-50)
    subsect_end = min(diff_coords.argmax()+50,len(diff_coords))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=start_coords[:,0], 
        y=start_coords[:,1], 
        z=start_coords[:,2],
        name='Start',
        #visible='legendonly',
        marker=dict(
            size=1,
            color='black',
        ),
        line=dict(
            color='black',
            width=10

        )
    ))
    fig.add_trace(go.Scatter3d(
        x=current_coords[:,0], 
        y=current_coords[:,1], 
        z=current_coords[:,2],
        name='Current',
        #visible='legendonly',
        marker=dict(
            size=1,
            color='blue',
        ),
        line=dict(
            color='blue',
            width=10
        )
    ))
    fig.add_trace(go.Scatter3d(
        x=current_coords[:,0], 
        y=current_coords[:,1], 
        z=current_coords[:,2],
        name='Change',
        visible='legendonly',
        marker=dict(
            size=1,
            color=diff_coords,
            colorscale='jet',
            colorbar=dict(thickness=20),
        ),
        line=dict(
            color=diff_coords,
            colorscale='jet',
            colorbar=dict(thickness=20),
            width=10
        )
    ))

    fig.update_layout(width=1000,height=1000)
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0),
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            aspectratio = dict( x=1, y=1, z=1 ),
            aspectmode = 'manual',
            xaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),
            yaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),
            zaxis = dict(
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                nticks=0,
                showticklabels=False),),
    )
    fig.show()

def write_curve_to_file(curve,outfile_name):
    with open(outfile_name,'w+') as f:
        for i in range(len(curve)-1):
            string = ' '.join(map(str,curve[i]))
            f.write(string)
            f.write('\n')
        f.write(' '.join(map(str,curve[-1])))
        f.close()