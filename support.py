import numpy as np
import matplotlib.pyplot as plt

class SupportFilesDrone:

    def __init__(self):


        # Constants
        Ixx = 0.0034 # kg*m^2
        Iyy = 0.0034 # kg*m^2
        Izz  = 0.006 # kg*m^2
        m  = 0.8 # kg
        g  = 9.81 # m/s^2
        Jtp=0.0000002342 # N*m*s^2=kg*m^2
        Ts=0.1 # s

        Q=np.matrix('10 0 0;0 10 0;0 0 10') # weights for outputs
        S=np.matrix('20 0 0;0 20 0;0 0 20') # weights for the final horizon period outputs
        R=np.matrix('10 0 0;0 10 0;0 0 10') # weights for inputs

        ct = 6.9471477491470765e-06 # N*s^2
        cq = 2.447423322999034e-07 # N*m*s^2
        l = 0.2 # m

        controlled_states=3 # Number of attitude outputs: Phi, Theta, Psi
        horizon = 6

        # The poles
        px=np.array([-0.5,-0.5])
        py=np.array([-0.5,-0.5])
        pz=np.array([-1,-1])

        r=3
        f=0.03
        hi=5
        hf=10

        self.constants=[Ixx, Iyy, Izz, m, g, Jtp, Ts, Q, S, R, ct, cq, l, controlled_states, horizon, px, py, pz, r, f, hi, hf]

        return None

    def trajectory(self,t):


        R=self.constants[18]
        f=self.constants[19]
        hi=self.constants[20]
        hf=self.constants[21]

        d=hf-hi


        alpha=2*np.pi*f*t

        x = R * np.cos(alpha)
        y = R * np.sin(alpha)
        z = hi + d / (t[-1]) * t

        x_dot = -R * np.sin(alpha) * 2 * np.pi * f
        y_dot = R* np.cos(alpha) * 2 * np.pi * f
        z_dot = d / (t[-1]) * np.ones(len(t))

        x_dot_dot = -R * np.cos(alpha) * (2 * np.pi * f) ** 2
        y_dot_dot = -R * np.sin(alpha) * (2 * np.pi * f) ** 2
        z_dot_dot = 0 * np.ones(len(t))

        dx=x[1:len(x)]-x[0:len(x)-1]
        dy=y[1:len(y)]-y[0:len(y)-1]


        dx=np.append(np.array(dx[0]),dx)
        dy=np.append(np.array(dy[0]),dy)


        psi=np.zeros(len(x))
        psiInt=psi
        psi[0]=np.arctan2(y[0],x[0])+np.pi/2
        psi[1:len(psi)]=np.arctan2(dy[1:len(dy)],dx[1:len(dx)])

        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psiInt[0]=psi[0]
        for i in range(1,len(psiInt)):
            if dpsi[i-1]<-np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]

        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, psiInt

    def linearisation(self,X_ref,X_dot_ref,X_dot_dot_ref,Y_ref,Y_dot_ref,Y_dot_dot_ref,Z_ref,Z_dot_ref,Z_dot_dot_ref,Psi_ref,states):


        m=self.constants[3]
        g=self.constants[4]
        px=self.constants[15]
        py=self.constants[16]
        pz=self.constants[17]

        u = states[0]
        v = states[1]
        w = states[2]
        x = states[6]
        y = states[7]
        z = states[8]
        phi = states[9]
        theta = states[10]
        psi = states[11]

        R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
        R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))
        pos_vel_body=np.array([[u],[v],[w]])
        pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]


        ex=X_ref-x
        ex_dot=X_dot_ref-x_dot
        ey=Y_ref-y
        ey_dot=Y_dot_ref-y_dot
        ez=Z_ref-z
        ez_dot=Z_dot_ref-z_dot


        kx1=(px[0]-(px[0]+px[1])/2)**2-(px[0]+px[1])**2/4
        kx2=px[0]+px[1]


        ky1=(py[0]-(py[0]+py[1])/2)**2-(py[0]+py[1])**2/4
        ky2=py[0]+py[1]


        kz1=(pz[0]-(pz[0]+pz[1])/2)**2-(pz[0]+pz[1])**2/4
        kz2=pz[0]+pz[1]


        ux=kx1*ex+kx2*ex_dot
        uy=ky1*ey+ky2*ey_dot
        uz=kz1*ez+kz2*ez_dot

        vx=X_dot_dot_ref-ux[0]
        vy=Y_dot_dot_ref-uy[0]
        vz=Z_dot_dot_ref-uz[0]


        a=vx/(vz+g)
        b=vy/(vz+g)
        c=np.cos(Psi_ref)
        d=np.sin(Psi_ref)
        tan_theta=a*c+b*d
        Theta_ref=np.arctan(tan_theta)

        if Psi_ref>=0:
            Psi_ref_singularity=Psi_ref-np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi
        else:
            Psi_ref_singularity=Psi_ref+np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi

        if ((np.abs(Psi_ref_singularity)<np.pi/4 or np.abs(Psi_ref_singularity)>7*np.pi/4) or (np.abs(Psi_ref_singularity)>3*np.pi/4 and np.abs(Psi_ref_singularity)<5*np.pi/4)):
            tan_phi=np.cos(Theta_ref)*(np.tan(Theta_ref)*d-b)/c
        else:
            tan_phi=np.cos(Theta_ref)*(a-np.tan(Theta_ref)*c)/d
        Phi_ref=np.arctan(tan_phi)
        U1=(vz+g)*m/(np.cos(Phi_ref)*np.cos(Theta_ref))

        return Phi_ref, Theta_ref, U1

    def discrete(self,states,omega_total):

        Ix=self.constants[0] # kg*m^2
        Iy=self.constants[1] # kg*m^2
        Iz=self.constants[2] # kg*m^2
        Jtp=self.constants[5] #N*m*s^2=kg*m^2
        Ts=self.constants[6] #s

        u=states[0]
        v=states[1]
        w=states[2]
        p=states[3]
        q=states[4]
        r=states[5]
        phi=states[9]
        theta=states[10]
        psi=states[11]

        R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
        R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))
        pos_vel_body=np.array([[u],[v],[w]])
        pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]
        x_dot=x_dot[0]
        y_dot=y_dot[0]
        z_dot=z_dot[0]

        T_matrix=np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
            [0,np.cos(phi),-np.sin(phi)],\
            [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])
        rot_vel_body=np.array([[p],[q],[r]])
        rot_vel_fixed=np.matmul(T_matrix,rot_vel_body)
        phi_dot=rot_vel_fixed[0]
        theta_dot=rot_vel_fixed[1]
        psi_dot=rot_vel_fixed[2]
        phi_dot=phi_dot[0]
        theta_dot=theta_dot[0]
        psi_dot=psi_dot[0]

        A01=1
        A13=-omega_total*Jtp/Ix
        A15=theta_dot*(Iy-Iz)/Ix
        A23=1
        A31=omega_total*Jtp/Iy
        A35=phi_dot*(Iz-Ix)/Iy
        A45=1
        A51=(theta_dot/2)*(Ix-Iy)/Iz
        A53=(phi_dot/2)*(Ix-Iy)/Iz

        A=np.zeros((6,6))
        B=np.zeros((6,3))
        C=np.zeros((3,6))
        D=0

        A[0,1]=A01
        A[1,3]=A13
        A[1,5]=A15
        A[2,3]=A23
        A[3,1]=A31
        A[3,5]=A35
        A[4,5]=A45
        A[5,1]=A51
        A[5,3]=A53

        B[1,0]=1/Ix
        B[3,1]=1/Iy
        B[5,2]=1/Iz

        C[0,0]=1
        C[1,2]=1
        C[2,4]=1

        D=np.zeros((3,3))

        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        Dd=D

        return Ad,Bd,Cd,Dd,x_dot,y_dot,z_dot,phi,phi_dot,theta,theta_dot,psi,psi_dot

    def mpc_matrice(self, Ad, Bd, Cd, Dd, hz):

        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)


        Q=self.constants[7]
        S=self.constants[8]
        R=self.constants[9]

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)


        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc

    def drone(self,states,omega_total,U1,U2,U3,U4):

        Ix=self.constants[0]
        Iy=self.constants[1]
        Iz=self.constants[2]
        m=self.constants[3]
        g=self.constants[4]
        Jtp=self.constants[5]
        Ts=self.constants[6]

        current_states=states
        new_states=current_states
        u = current_states[0]
        v = current_states[1]
        w = current_states[2]
        p = current_states[3]
        q = current_states[4]
        r = current_states[5]
        x = current_states[6]
        y = current_states[7]
        z = current_states[8]
        phi = current_states[9]
        theta = current_states[10]
        psi = current_states[11]

        u_dot = (v * r - w * q) + g * np.sin(theta)
        v_dot = (w * p - u * r) - g * np.cos(theta) * np.sin(phi)
        w_dot = (u * q - v * p) - g * np.cos(theta) * np.cos(phi) + U1 / m
        p_dot = q * r * (Iy - Iz) / Ix - Jtp / Ix * q * omega_total + U2 / Ix
        q_dot = p * r * (Iz - Ix) / Iy + Jtp / Iy * p * omega_total + U3 / Iy
        r_dot = p * q * (Ix - Iy) / Iz + U4 / Iz

        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        R = np.matmul(R_z, np.matmul(R_y, R_x))

        T= np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)], \
                             [0, np.cos(phi), -np.sin(phi)], \
                             [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])


        u = u + Ts * u_dot  # u,v,w are the linear velocities of the body in body referential
        v = v + Ts * v_dot
        w = w + Ts * w_dot

        p = p + Ts * p_dot
        q = q + Ts * q_dot  # p,q,r are the rotational velocities of the body in body referential
        r = r + Ts * r_dot

        firstu = np.array([u, v, w])
        firstp = np.array([p, q, r])

        refstatepointx = np.matmul(R, firstu)

        refstatepointphi = np.matmul(T, firstp)
        x = x + refstatepointx[0] * Ts
        y = y + refstatepointx[1] * Ts
        z = z + refstatepointx[2] * Ts

        phi = phi + refstatepointphi[0] * Ts
        theta = theta + refstatepointphi[1] * Ts
        psi = psi + refstatepointphi[2] * Ts
        new_states[0]=u
        new_states[1]=v
        new_states[2]=w
        new_states[3]=p
        new_states[4]=q
        new_states[5]=r

        new_states[6]=x
        new_states[7]=y
        new_states[8]=z
        new_states[9]=phi
        new_states[10]=theta
        new_states[11]=psi

        return new_states
