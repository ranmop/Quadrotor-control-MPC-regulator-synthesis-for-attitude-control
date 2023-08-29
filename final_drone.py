from roblib import *

import platform

import numpy as np

import support as sfd

fig = plt.figure()

ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(322)
ax2 = fig.add_subplot(324)
ax3 = fig.add_subplot(326)
ax1.grid()
ax2.grid()
ax3.grid()


ech=25
support=sfd.SupportFilesDrone()
constants=support.constants
ar=support.constants[18]
height_f=support.constants[21]

def draw_quadri(X):
   ax0.clear()
   ax0.set_xlim3d(-ar,ar)
   ax0.set_ylim3d(-ar,ar)
   ax0.set_zlim3d(0,height_f)
   ax0.set_xlabel('$X$', fontsize=15)
   ax0.set_ylabel('$Y$',fontsize=15)
   ax0.set_zlabel('$Z$', fontsize=15)
   l=1
   draw_quadrotor3D(ax0,X,a,0.5*l)


a=array([[9,0,0,0]]).T



Ts=constants[6]
controlled_states=constants[13]
innerDyn_length=constants[14]

t=np.arange(0,100+Ts*innerDyn_length,Ts*innerDyn_length) # time from 0 to 100 seconds, sample time Ts=0.4 second

X_ref,X_dot_ref,X_dot_dot_ref,Y_ref,Y_dot_ref,Y_dot_dot_ref,Z_ref,Z_dot_ref,Z_dot_dot_ref,psi_ref=support.trajectory(t)
plotl=len(t) # Number of outer control loop iterations

# Load the initial state vector
ut=0
vt=0
wt=0
pt=0
qt=0
rt=0
xt=0
yt=0
zt=0
phit=0
thetat=0
psit=psi_ref[0]

states=np.array([ut,vt,wt
                    ,pt,qt,rt,xt,yt,zt,phit,thetat,psit])
statesTotal=[states]

statesTotal_ani=[states[6:len(states)]]

ref_angles_total=np.array([[phit,thetat,psit]])

velocityXYZ_total=np.array([[0,0,0]])

omega1=20000
omega2=20000
omega3=20000
omega4=20000
omega_total=omega1-omega2+omega3-omega4

ct=constants[10]
cq=constants[11]
l=constants[12]

U1=ct*(omega1**2+omega2**2+omega3**2+omega4**2) # Input at t = -Ts s
U2=ct*l*(omega2**2-omega4**2) # Input at t = -Ts s
U3=ct*l*(omega3**2-omega1**2) # Input at t = -Ts s
U4=cq*(-omega1**2+omega2**2-omega3**2+omega4**2) # Input at t = -Ts s
UTotal=np.array([[U1,U2,U3,U4]]) # 4 inputs

a = array([[0, 0, 0, 0]]).T
trajx =np.array(0)
trajy =np.array(0)
trajz =np.array(0)
time=np.array(0)
########## Start  #################################

for il in range(0,plotl-1):
    # Implement the position controller (state feedback linearization)
    phi_ref, theta_ref, U1=support.linearisation(X_ref[il+1],X_dot_ref[il+1],X_dot_dot_ref[il+1],Y_ref[il+1],Y_dot_ref[il+1],Y_dot_dot_ref[il+1],Z_ref[il+1],Z_dot_ref[il+1],Z_dot_dot_ref[il+1],psi_ref[il+1],states)
    Phi_ref=np.transpose([phi_ref*np.ones(innerDyn_length+1)])
    Theta_ref=np.transpose([theta_ref*np.ones(innerDyn_length+1)])
    Psi_ref=np.transpose([psi_ref[il+1]*np.ones(innerDyn_length+1)])

    temp_angles=np.concatenate((Phi_ref[1:len(Phi_ref)],Theta_ref[1:len(Theta_ref)],Psi_ref[1:len(Psi_ref)]),axis=1)
    ref_angles_total=np.concatenate((ref_angles_total,temp_angles),axis=0)
    # Create a reference vector
    refSignals=np.zeros(len(Phi_ref)*controlled_states)

    # Build up the reference signal vector:

    k=0
    for i in range(0,len(refSignals),controlled_states):
        refSignals[i]=Phi_ref[k]
        refSignals[i+1]=Theta_ref[k]
        refSignals[i+2]=Psi_ref[k]
        k=k+1

    #
    hz=support.constants[14]
    k=0

    for i in range(0,innerDyn_length):
        # Generate the discrete state space matrices
        Ad,Bd,Cd,Dd,x_dot,y_dot,z_dot,phi,phi_dot,theta,theta_dot,psi,psi_dot=support.discrete(states, omega_total)
        x_dot=np.transpose([x_dot])
        y_dot=np.transpose([y_dot])
        z_dot=np.transpose([z_dot])
        temp_velocityXYZ=np.concatenate(([[x_dot],[y_dot],[z_dot]]),axis=1)
        velocityXYZ_total=np.concatenate((velocityXYZ_total,temp_velocityXYZ),axis=0)

        x_aug_t=np.transpose([np.concatenate(([phi,phi_dot,theta,theta_dot,psi,psi_dot],[U2,U3,U4]),axis=0)])
        k=k+controlled_states
        if k+controlled_states*hz<=len(refSignals):
            r=refSignals[k:k+controlled_states*hz]
        else:
            r=refSignals[k:len(refSignals)]
            hz=hz-1

        Hdb,Fdbt,Cdb,Adc=support.mpc_matrice(Ad,Bd,Cd,Dd,hz)
        ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0),Fdbt)

        du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))

        U2=U2+du[0][0]
        U3=U3+du[1][0]
        U4=U4+du[2][0]

        UTotal = np.concatenate((UTotal, np.array([[U1, U2, U3, U4]])), axis=0)

        U1C = U1 / ct
        U2C = U2 / (ct * l)
        U3C = U3 / (ct * l)
        U4C = U4 / cq

        UC_vector = np.zeros((4, 1))
        UC_vector[0, 0] = U1C
        UC_vector[1, 0] = U2C
        UC_vector[2, 0] = U3C
        UC_vector[3, 0] = U4C

        omega_Matrix = np.zeros((4, 4))
        omega_Matrix[0, 0] = 1
        omega_Matrix[0, 1] = 1
        omega_Matrix[0, 2] = 1
        omega_Matrix[0, 3] = 1
        omega_Matrix[1, 1] = 1
        omega_Matrix[1, 3] = -1
        omega_Matrix[2, 0] = -1
        omega_Matrix[2, 2] = 1
        omega_Matrix[3, 0] = -1
        omega_Matrix[3, 1] = 1
        omega_Matrix[3, 2] = -1
        omega_Matrix[3, 3] = 1

        omega_Matrix_inverse = np.linalg.inv(omega_Matrix)
        omegas_vector = np.matmul(omega_Matrix_inverse, UC_vector)

        omega1P2 = omegas_vector[0, 0]
        omega2P2 = omegas_vector[1, 0]
        omega3P2 = omegas_vector[2, 0]
        omega4P2 = omegas_vector[3, 0]

        if omega1P2 <= 0 or omega2P2 <= 0 or omega3P2 <= 0 or omega4P2 <= 0:
            print("You can't take a square root of a negative number")
            exit()
        else:
            omega1 = np.sqrt(omega1P2)
            omega2 = np.sqrt(omega2P2)
            omega3 = np.sqrt(omega3P2)
            omega4 = np.sqrt(omega4P2)

        omega_total = omega1 - omega2 + omega3 - omega4

        states= support.drone(states, omega_total, U1, U2, U3, U4)

        x = states[6]
        y = states[7]
        z = states[8]
        phi = states[9]
        theta = states[10]
        psi = states[11]
        trajx=np.append(trajx,x)
        trajy =np.append(trajy,y)
        trajz =np.append(trajz,z)
        time = np.append(time, il)
        draw_quadri(-1 * array([[x, -y, z, phi, theta, psi, 10, 10, 0, 0, 0, 0]]).T)
        ref_trajectory = ax0.plot(X_ref, Y_ref, Z_ref, 'b', linewidth=1, label='reference')
        drone_trajectory=ax0.plot(trajx,trajy,trajz,'r')
        draw_xr=ax1.plot(X_ref,'b',linewidth=1)
        draw_xdrone=ax1.plot(time,trajx,'r',linewidth=1)
        draw_yr = ax2.plot(Y_ref, 'b', linewidth=1)
        draw_ydrone = ax2.plot(time, trajy, 'r', linewidth=1)
        draw_zr = ax3.plot(Z_ref, 'b', linewidth=1)
        draw_zdrone = ax3.plot(time, trajz, 'r', linewidth=1)
        pause(0.01)

plt.show()