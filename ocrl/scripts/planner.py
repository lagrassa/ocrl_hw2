from common import *
import scipy.optimize as opt
import autograd.numpy as np
from autograd import grad,jacobian
from matplotlib import pyplot as plt
global debug
debug = False

# from https://github.com/gerdl/scc_paths
from scc.turn import Turn
from scc.turnparams import TurnParams
from scc.state import State
from scc.scc_path_variant import SccPathVariant,PathType


def split_state(x):
    n = (x.shape[0]-1)//4
    tk = x[0:n+1]
    xk = x[n+1:2*n+1]
    yk = x[2*n+1:n*3+1]
    thetak = x[n*3+1:]

    return tk, xk, yk, thetak

def sign(v):
    tol = 1e-5
    k = 5e3
    res = np.where(np.fabs(v) <= tol,0, np.tanh(k*v))
    return res


# nonlinear (positive) constraints
def get_c(x):
    n = (x.shape[0]-1)//4
    tk, xk, yk, thetak = split_state(x)


    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    dx = xk[1:] - xk[0:-1]
    dy = yk[1:] - yk[0:-1]
    dtheta =  thetak[1:] - thetak[0:-1]
    dk = np.array([dx,dy,np.zeros(n+1)]).T

    # turning radius

    mask = np.fabs(dtheta) > 1e-5
    dtheta_stable = np.where(mask,dtheta,1)
    turn_rad = np.where(mask,np.linalg.norm(dk,axis=1)/np.fabs(2*np.sin(dtheta_stable/2)),min_turning_radius)
    c_rad = turn_rad - min_turning_radius

    # linear velocity
    qk = np.array([np.cos(thetak[:-1]),np.sin(thetak[:-1]),np.zeros(n+1)]).T
    proj_q_d = np.sum(qk * dk, axis=1)
    sign_v = sign(proj_q_d)
    vk = np.linalg.norm(dk,axis=1)*sign_v/tk
    c_vel = 0.9*max_vel - np.fabs(vk)

    # angular velocity
    wk = dtheta/tk
    c_w = 0.8*max_ang_vel - np.fabs(wk)

    vk = np.concatenate(([start_vel],vk))
    # acceleration (finite differences)
    ak = (vk[1:] - vk[0:-1])/tk

    c_acc = np.where(ak<0, ak + 0.5*max_dec,0.5*max_acc - ak)
    c = np.concatenate((c_rad,c_vel,c_w,c_acc))
    return c


# equality (=0) constraints
def get_ceq(x):
    n = (x.shape[0]-1)//4

    tk, xk, yk, thetak = split_state(x)
    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    dx = xk[1:] - xk[0:-1]
    dy = yk[1:] - yk[0:-1]
    dk = np.array([dx,dy,np.zeros(n+1)]).T

    # constant curvature (trapezoidal collocation, see teb paper)
    ceq = np.array([np.cos(thetak[0:-1]) + np.cos(thetak[1:]),np.sin(thetak[0:-1]) + np.sin(thetak[1:]),np.zeros(n+1)]).T
    ceq = np.fabs(np.cross(ceq,dk,axisa=1,axisb=1)[:,2])
    return ceq


def obj_func(x):
    n = (x.shape[0]-1)//4
    tk, xk, yk, thetak = split_state(x)
    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    time_cost =  np.sum(tk)

    # constant curvature path constraint
    path_cost = np.fabs(get_ceq(x))**2
    # vel, acc, radius limits
    ineq_cost = np.fabs(np.minimum(0,get_c(x)))**2

    # discourage sharp turns (a bit hacky but wanted paths to be less jagged)
    dist = (xk[1:] - xk[:-1])**2 + (yk[1:] - yk[:-1])**2

    # teb paper said that path cost should be much larger than the rest
    # but these weights/entire cost function could probably use some tuning
    cost = .01*time_cost + np.sum(5*path_cost) + 5*np.sum(ineq_cost)  + .1*np.sum(dist)


    # cost = 0.001*time_cost + np.sum( 5000*path_cost) + 10*np.sum(ineq_cost)  + 100*np.sum(dist)

    return cost

def lm_cost(x):
    n = (x.shape[0]-1)//4
    tk, xk, yk, thetak = split_state(x)
    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    time_cost =  np.sum(tk)

    # constant curvature path constraint
    path_cost = np.fabs(get_ceq(x))
    # vel, acc, radius limits
    ineq_cost = np.fabs(np.minimum(0,get_c(x)))

    # discourage sharp turns (a bit hacky but wanted paths to be less jagged)
    dist = (xk[1:] - xk[:-1])**2 + (yk[1:] - yk[:-1])**2

    # teb paper said that path cost should be much larger than the rest
    # but these weights/entire cost function could probably use some tuning
    # cost = .1*time_cost + np.sum(5*path_cost) + 5*np.sum(ineq_cost)  + .1*np.sum(dist)
    cost = np.concatenate(([0.01*time_cost],10*path_cost,10*ineq_cost,0.1*dist))

    # cost = 0.001*time_cost + np.sum( 5000*path_cost) + 10*np.sum(ineq_cost)  + 100*np.sum(dist)

    return cost

def get_bounds(nc):
    max_time = 10
    bounds = np.zeros(((nc-1)*3 + 1,2))

    bounds[0,:] = [1e-5*nc,max_time]
    bounds[1:nc,:] = 0.95*np.array(x_lim)
    bounds[nc:2*nc,:] = 0.95*np.array(y_lim)
    bounds[2*nc:3*nc,:] = theta_lim
    return bounds


def compute_control_inputs(fc,xk,yk,thetak):
    n = xk.shape[0]-1
    # import ipdb; ipdb.set_trace()

    dx = xk[1:] - xk[0:-1]
    dy = yk[1:] - yk[0:-1]
    dk = np.array([dx,dy,np.zeros(n)]).T
    dtheta =  (thetak[1:] - thetak[0:-1])/(1.0/fc)

    # vel
    qk = np.array([np.cos(thetak[:-1]),np.sin(thetak[:-1]),np.zeros(n)]).T
    proj_q_d =  np.sum(qk * dk, axis=1)
    sign_v = sign(proj_q_d)
    vk = np.linalg.norm(dk,axis=1)*sign_v/(1.0/fc)

    # steering angle
    phi_k = np.zeros(vk.shape)
    mask = np.absolute(vk) > 1e-5
    phi_k[mask] = np.arctan(wheelbase*dtheta[mask]/vk[mask])
    u = np.array((vk,phi_k))
    return u

def resize_traj(tc, tk,xk,yk,thetak):
    modified = True
    i = 0

    while (modified and i < 100):
        modified = False
        n = 0
        while n < tk.shape[0]:
            if (tk[n] > tc*1.1) and (tk.shape[0] < 300):
                new_dt = 0.5*tk[n]
                tk[n] = new_dt
                new_x = 0.5*(xk[n+1] + xk[n])
                new_y = 0.5*(yk[n+1] + yk[n])
                new_theta = avg_angle(thetak[n],thetak[n+1])
                tk = np.insert(tk,n,new_dt)
                xk = np.insert(xk,n+1,new_x)
                yk = np.insert(yk, n+1, new_y)
                thetak = np.insert(thetak, n+1, new_theta)
                modified = True
                n+=1
            n += 1
        i += 1
    return tk,xk,yk,thetak


def avg_angle(th1,th2):
    x = np.cos(th1) + np.cos(th2)
    y = np.sin(th1) + np.sin(th2)
    if (x == 0 and y == 0):
        return 0
    else:
        return np.arctan2(y, x)

def plan(s0,sf,v0,fc):
    tc = 1.0/fc

    global start_x,start_y, start_theta, start_vel, goal_theta,goal_x, goal_y, goal_theta
    start_vel = v0
    start_x = s0[0]
    start_y = s0[1]
    start_theta = s0[2]
    goal_x = sf[0]
    goal_y = sf[1]
    goal_theta = sf[2]

    time_traj = 5

    dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)

    import dubins
    path = dubins.shortest_path(s0,sf,min_turning_radius)
    dist = path.path_length()
    t0 = dist/(0.9*max_vel)
    nc = int(t0/tc)
    points = path.sample_many(dist/nc)[0]
    init_x = np.array(points)[:,0]
    init_y = np.array(points)[:,1]
    init_theta = np.array(points)[:,2]
    # set parameters:
    # pos1 = State(_x=start_x, _y=start_y, _theta=start_theta, _kappa=0)
    # pos2 = State(_x=goal_x, _y=goal_y, _theta=goal_theta, _kappa=0)
    # tparam = TurnParams(_kappa_max=1/min_turning_radius,
    #                     _sigma_max=min(1,dist/min_turning_radius))
    #
    # PATHOPTIONS = [PathType.lsl, PathType.rsr, PathType.rsl, PathType.lsr]
    # path = SccPathVariant(tparam, pos1, pos2, PathType.rsl)
    # paths = []
    # valid = True
    # for variant in PATHOPTIONS:
    #     paths.append(SccPathVariant(tparam, pos1, pos2, variant))
    #     if not paths[-1].valid:
    #         not_valid = False
    #         break
    #
    # if valid:
    #     shortest_path = min(paths, key=lambda path: path.len)
    #
    #     # calculate positions:
    #     X = np.linspace(0, shortest_path.len, nc+1, endpoint=True)
    #     tra = shortest_path.state(X)
    #     # Optimization is very sensitive to initialization
    #     init_x = tra.x
    #     init_y = tra.y
    #     tra.theta[tra.theta > np.pi] = tra.theta[tra.theta > np.pi]  - 2*np.pi
    #     init_theta = tra.theta
    # else:
    #     print("WARNING: initial curved path was invalid. using linear path for initialization instead.")
    #
    #     init_x = np.linspace(start_x,goal_x,nc+1)
    #     init_y = np.linspace(start_y,goal_y,nc+1)
    #     init_theta = np.linspace(start_theta,goal_theta,nc+1)

    plt.plot(init_x,init_y)




    ineq_constr = {'type': 'ineq',
                   'fun': get_c,
                   'jac':  jacobian(get_c)}
    eq_constr = {'type':'eq',
                 'fun': get_ceq,
                 'jac': jacobian(get_ceq)}
    bounds = get_bounds(nc)
    options = {'ftol':1e-9,'disp':True,'iprint':2}
    jac = grad(obj_func)

    def jac_reg(x):
        j = jac(x)
        if np.isfinite(j).all():
            return j
        else:
            return opt.approx_fprime(x0,obj_func,1e-6)

    jac_lm = jacobian(lm_cost)

    def jac_reg_lm(x):
        j = jac_lm(x)
        if np.isfinite(j).all():
            return j
        else:
            return opt.approx_fprime(x0,lm_cost,1e-6)
    # not sure which method words best. both kind of suck
    # import ipdb; ipdb.set_trace()
    #res = opt.minimize(obj_func,x0,method='SLSQP',jac=jac_reg,constraints=[eq_constr,ineq_constr],options=options,bounds=bounds)
    init_t = [t0/(nc)]*nc
    for i in range(0,2):
        x0 = np.concatenate((init_t,init_x[1:-1],init_y[1:-1],init_theta[1:-1]))

        res = opt.least_squares(lm_cost,x0,method="lm",ftol=1e-8,xtol=1e-8,jac=jac_reg_lm,verbose=2,max_nfev=50) # bounds = (bounds[:,0],bounds[:,1])
        tk, xk, yk, thetak = split_state(res.x)
        xk = np.concatenate(([start_x], xk, [goal_x]))
        yk = np.concatenate(([start_y], yk, [goal_y]))
        thetak = np.concatenate(([start_theta], thetak, [goal_theta]))

        tk,xk,yk,thetak = resize_traj(tc,tk,xk,yk,thetak)
        init_x = xk
        init_y = yk
        init_theta = thetak
        init_t = tk



    if True:

        print("goal:" + str(sf[0])+ " " + str(sf[1])+ " " + str(sf[2]))
        print("res:" + str(xk[-1])+ " " + str(yk[-1])+ " " + str(thetak[-1]))

        plt.plot(xk,yk,'b',xk[-1],yk[-1],'g*')
        plt.show()

    # control frequency
    # compute control inputs
    uc = compute_control_inputs(fc,xk,yk,thetak)
    if debug:
        ax = plt.subplot(1,1,1)
        ax.plot(uc[0],uc[1])
        plt.show()

    print("end distance", np.linalg.norm(np.array([xk[-1], yk[-1], thetak[-1]])-sf))
    return uc, np.array([xk,yk,thetak])


# Test Script
if __name__ == '__main__':
    global debug
    debug = False
    # Generate random waypoints
    waypoints = np.random.rand(num_waypoints, 3)
    waypoints[:, 0] = (x_lim[1] - x_lim[0]) * waypoints[:, 0] + x_lim[0]
    waypoints[:, 1] = (y_lim[1] - y_lim[0]) * waypoints[:, 1] + y_lim[0]
    waypoints[:, 2] = (theta_lim[1] - theta_lim[0]) * waypoints[:, 2] + theta_lim[0]
    
    fc = 30
    s0 = np.array([0,0,0])
    v0 = 0

    for i in range(0,num_waypoints):
        sf = waypoints[i, :]
        uc, wp_out = plan(s0, sf,v0,fc)
        v0 = uc[0,-1]
        s0 = wp_out[:, -1]
