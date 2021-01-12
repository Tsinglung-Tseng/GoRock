import numpy as np
import matplotlib.pyplot as plt

def cylinder_source(ind, centre=(0,0), radius=1., activity=0.):
    x = np.array(centre, dtype=np.float32)[0]
    z = np.array(centre, dtype=np.float32)[1]
    def _leave_2_dec(a):
        return str(a).split('.')[0] + '.' + str(a).split('.')[1][:2]
    return f"""/gate/source/addSource                            Mysource{ind}
/gate/source/Mysource{ind}/gps/particle               gamma
/gate/source/Mysource{ind}/gps/energytype             Mono
/gate/source/Mysource{ind}/gps/angtype                iso
/gate/source/Mysource{ind}/setType                    backtoback
/gate/source/Mysource{ind}/setForcedUnstableFlag      true
/gate/source/Mysource{ind}/setForcedHalfLife          6586 s
/gate/source/Mysource{ind}/gps/type           Volume
/gate/source/Mysource{ind}/gps/shape          Cylinder
/gate/source/Mysource{ind}/gps/radius         {_leave_2_dec(str(radius))} mm
/gate/source/Mysource{ind}/gps/halfz          0.25 mm
/gate/source/Mysource{ind}/setActivity        {int(activity)} becquerel
/gate/source/Mysource{ind}/gps/monoenergy             511. keV
/gate/source/Mysource{ind}/gps/polarization           1. 0. 0.
/gate/source/Mysource{ind}/gps/centre                 {_leave_2_dec(str(x))} 0.0 {_leave_2_dec(str(z))} mm
/gate/source/Mysource{ind}/gps/posrot1   1 0 0
/gate/source/Mysource{ind}/gps/posrot2   0 0 1"""


def sphere_vol(r):
    return (4/3)*np.pi*r*r*r

def cylinder_vol(r):
    h = 0.5
    return h*np.pi*r*r

def dose(r, shape_vol):
    "Accept vol calculate procedure."
    BASE_DOSE=500
    unit_vol_dose = BASE_DOSE*shape_vol(1)
    return unit_vol_dose*r

def d_phan_section(radius, num_layers):
    num_layers = int(num_layers)
    SECTION_DISTANCE_TO_CENTER = 7.5
    
    def radius_to_delta():
        return radius*4, np.sqrt(3)*radius*2
    def nth_layer_deltas(nth_layer):
        delta_xs = np.array(range(nth_layer))*delta_x - (nth_layer-1)/2*delta_x
        delta_ys = np.full_like(delta_xs, (nth_layer-1)*delta_y) + SECTION_DISTANCE_TO_CENTER
        return np.vstack([delta_xs, delta_ys]).T
    
    delta_x, delta_y = radius_to_delta()
    nth_layer_coord = np.vstack([nth_layer_deltas(i).tolist() for i in range(1, num_layers+1)])

    return np.vstack([
        nth_layer_coord.T, 
        np.full_like(nth_layer_coord[:,0], radius)
    ]).T 

def rotate_2d(point, theta):
    return np.matmul(point, [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

STANDARD_PHANTOM = np.array([
    [0.4, 8],
    [1.25, 3],
    [1., 4],
    [0.75, 5],
    [0.625, 6],
    [0.5, 7]
])

STANDARD_PHANTOM = np.vstack([STANDARD_PHANTOM.T, np.array(range(6))*np.pi/3]).T


x_result = np.vstack([
    np.concatenate([
        rotate_2d(
            d_phan_section(radius, num_layer)[:,:2],
            angle
        ),
        d_phan_section(radius, num_layer)[:,2:],
        np.full_like(d_phan_section(radius, num_layer)[:,2:], dose(radius, cylinder_vol))
    ], axis=1)
    for radius, num_layer, angle in STANDARD_PHANTOM
])
x_result = np.hstack([np.reshape(np.array(range(1, x_result.shape[0]+1)), [x_result.shape[0], 1]), x_result])


plt.scatter(x_result[:,1], x_result[:,2], s=x_result[:,3]*40)
plt.axis('equal')

print('\n'.join([cylinder_source(int(ind), [x, y], radius, activity) for ind, x, y, radius, activity in x_result]))
