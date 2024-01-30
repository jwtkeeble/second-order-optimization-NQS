import os, sys
# from termios import VT0
# from turtle import forward
import torch
from torch import Tensor, nn
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DIR='./'
torch.set_printoptions(6)
torch.set_default_dtype(torch.float32)

class HFsolver(nn.Module):
    
    def __init__(self, A:int=2, xL:int=6, Nx:int=240, V0:float=0., s:float=0.5, device:torch.device=torch.device('cpu'), itermax: int=20000) -> None:
        super(HFsolver, self).__init__()

        #interaction strength and range (hbar-omega and ho units)
        self.V0 = V0
        self.s = s

        #Real-Space mesh dimension and number of points
        self.xL=xL
        self.Nx=Nx

        self.device = device

        #Mesh in x-space
        self.delx = 2*xL/Nx
        self.xx = self.delx*(torch.arange(self.Nx,device=self.device)-self.Nx/2.)
        indx_origin = int(self.Nx/2)
        self.x1, self.x2 = torch.meshgrid([self.xx,self.xx], indexing='ij')

        #Mesh in p-space
        self.delp = 2.*math.pi/(2.*self.xL)
        self.pp = self.delp*(torch.arange(self.Nx,device=self.device)-self.Nx/2.)

        #2nd-derivative matrix
        self.cder2 = torch.zeros((Nx,Nx), device=self.device, dtype=torch.complex64)
        self.der2 = torch.zeros((Nx, Nx), device=self.device, dtype=torch.float32)

        for i, xi in enumerate(self.xx):
            for j, xj in enumerate(self.xx):
                self.cder2[i,j] = torch.exp(1j * (xj - xi)*self.pp) @ self.pp.cfloat()**2

        self.der2 = -1.*self.cder2.real*self.delx*self.delp/2./math.pi
        self.kin_mat = -self.der2/2.

        self.U_HO = self.xx**2/2.

        #Hartree-Fock Potential
        self.accu=1e-8
        self.itermax = itermax #20000
        self.pfac=1./(self.kin_mat.amax().abs())
        self.ffac=0.4

        self.hf_den = torch.zeros(self.Nx,device=self.device)
        self.denf = torch.zeros(self.Nx,device=self.device)
        self.Uex = torch.zeros(self.Nx,device=self.device)
        self.hf_den_mat = torch.zeros((self.Nx, self.Nx), device=self.device)
        self.H = torch.zeros((self.Nx, self.Nx), device=self.device)

        self.Nmax = A #A_num_part
        self.spe = torch.zeros(self.Nmax, device=self.device)
        self.wfy = torch.stack([self.wfho(ieig, self.xx) for ieig in range(self.Nmax)],dim=1)

        self.Vint = self.V0 * (2.*math.pi)**(-0.5) / self.s * torch.exp( -(self.x1-self.x2)**2/2./self.s**2 )

    def wfho(self, n: int, x: Tensor) -> Tensor:

        if(n>100 or n<0):
            raise ValueError("n value is wrong!")
        
        nfac = math.factorial(n) #log for stability?
        norm = (2**n * nfac)**(-0.5) / math.pi**0.25
    
        if(n==0):
            Hn = torch.ones_like(x)
        elif(n==1):
            Hn = 2.*x
        else:
            hm2 = torch.ones_like(x)
            hm1 = 2.*x
            for m in range(2,n+1):
                hmx = 2.*x*hm1 - 2.*np.real(m-1)*hm2 #was np.real(m-1) for some reason...?
                hm2=hm1.clone()
                hm1=hmx.clone()
            Hn = hmx
        
        wf = norm*torch.exp(-x.pow(2)/2.)*Hn
        return wf

    def forward(self):
        iter=0
        diffs=10.
        #header_screen = "%04s %012s %012s %012s %012s %012s %012s %012s %012s" % ("# ITER", "NUM_PART", "X_CM", "EHF ", "EHF2 ", "EKIN ", "EPOT ", "ESUM ", "DIFFS")
        #print(header_screen)
        #while(diffs>self.accu and iter<self.itermax):
        self.esum0hf_array = np.zeros(self.itermax+1)
        for iter in tqdm(range(0, self.itermax), desc=f'V0: {self.V0:3} s: {self.s:4.2f}'):
            iter+=1
            self.hf_den.zero_()
            self.hf_den_mat.zero_()
            for ieig in range(self.Nmax):
                self.hf_den += self.wfy[:,ieig].abs().pow(2)
                self.hf_den_mat += torch.einsum("i,j->ij", self.wfy[:,ieig], self.wfy[:,ieig])
            
            self.denf = self.hf_den.clone()                  #compute mean field
            self.Udir = self.delx * self.Vint @ self.denf    #direct term

            Umf_mat = -self.delx*self.Vint*self.hf_den_mat   #exchange term

            self.Uexc = Umf_mat.diagonal().clone()
            Umf_mat[range(self.Nx), range(self.Nx)] = Umf_mat.diagonal() + self.Udir + self.U_HO
            self.Umf = Umf_mat.diagonal().clone()

            H = self.kin_mat + Umf_mat
            diffs=0.
            ekin0hf=0.
            epot0hf=0.
            for ieig in range(self.Nmax):
                wf0 = self.wfy[:,ieig]
                wff=H@wf0
                self.spe[ieig]=(wf0@wff).real * self.delx

                wfb = wf0 - wff*self.pfac
                norm = wfb.abs().pow(2).sum()*self.delx
                wff=wfb/norm.sqrt()

                wfo=0.
                for jeig in range(0,ieig):
                    wfo += self.wfy[:,jeig]*(self.wfy[:,jeig]@wff)*self.delx
                wff-=wfo
                norm=wff.abs().pow(2).sum()*self.delx
                wff/=norm

                diffs += (wf0-wff).abs().amax()
                self.wfy[:,ieig] = self.ffac*wff + (1.-self.ffac)*wf0

                if(ieig <= self.Nmax):
                    ekin0hf += (self.wfy[:,ieig] @ (self.kin_mat @ self.wfy[:,ieig])).real * self.delx
                    epot0hf += (self.wfy[:,ieig] @ (Umf_mat @ self.wfy[:,ieig])).real * self.delx

            esum0hf = self.spe.sum()
            self.esum0hf_array[iter] = esum0hf.detach().numpy()
            #1st HF Determination using Kinetic Energy
            xa = torch.sum(self.hf_den) * self.delx
            x2_av = self.delx*( self.hf_den @ self.xx.pow(2) ) / (self.delx*self.hf_den.sum())

            #Energy from GMK Sum rule
            eho = torch.sum(self.hf_den*self.U_HO) * self.delx/2.
            enerhfp = (esum0hf + ekin0hf)/2. + eho

            #Energy from integral (DF)
            epot0hf = epot0hf-2.*eho
            enerhf=esum0hf-epot0hf/2.
            #if(iter%10000==0):
            #    print(f"{iter:6} {xa:12.8f} {x2_av:12.8f} {enerhf:12.8f} {enerhfp:12.8f} {ekin0hf:12.8f} {epot0hf:12.8f} {esum0hf:12.8f} {diffs:12.8f}")
        return enerhf, enerhfp, ekin0hf, eho, epot0hf, esum0hf
