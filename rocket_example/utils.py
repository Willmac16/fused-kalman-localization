import csv
import numpy as np
import noise # pip install noise

def calculate_air_density(altitude):
    # Simplified model of air density as a function of altitude
    # This can be replaced with a more accurate model if needed
    return 1.225 * np.exp(-altitude / 8500)  # Exponential decay of density with altitude

def calculate_drag(velocity, air_density, C_d, A_s):
    v_mag = np.linalg.norm(velocity)
    return 0.5 * air_density * v_mag**2 * C_d * A_s

def wind_vector(x, y, z):
    # Generate wind vector using Perlin noise
    return np.array([noise.pnoise3(x/400, y/400, z/400, octaves=2, persistence=0.01), 
                     noise.pnoise3(x/400 + 100, y/400 + 100, z/400+ 100, octaves=2, persistence=0.01), 
                     noise.pnoise3(x/400 + 200, y/400 + 200, z/400 + 200, octaves=2, persistence=0.01)])
    
def read_thrust():
    # Read in the Cesaroni thrust data from the CSV file
    thrust_data = []
    with open('Cesaroni_21062O3400-P.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            thrust_data.append(row)
    #print(len(thrust_data))

    # Extract the time and thrust values from the data
    time = [float(row[0]) for row in thrust_data[1:]]
    thrust = [float(row[1]) for row in thrust_data[1:]]
    time.insert(0, 0)
    thrust.insert(0, 0)
    return time, thrust


def f_thrust(t):
    time, thrust = read_thrust()
    if t < 0 or t > time[-1]:
        return 0
    else:
        lerp_thrust = np.interp(t, time, thrust)
        return lerp_thrust
    
    
def mass(t):
    time, thrust = read_thrust()
    if t > time[-1]:
        return 0
    total_mass = 16.84  # [kg]
    total_thrust = np.trapz(thrust, time)
    times = np.linspace(0, t, 30)
    thrusts = [f_thrust(t) for t in times]
    return total_mass - np.trapz(thrusts, times) / total_thrust * total_mass


def cg(t, dry_mass=18.88):
    """Returns the center of gravity of the rocket at time t from the tip

    Args:
        t (float): the time in sectonds
        dry_mass (float, optional): the dry mass of the rocket. Defaults to 18.88.

    Returns:
        float: the cg of the rocket at time t
    """
    rocket_offset = 1
    motor_offset = 2
    wet_mass = mass(t)
    com_vect = (wet_mass * motor_offset + dry_mass * rocket_offset)  / (wet_mass + dry_mass)
    return com_vect




def rot_by_q(q,v):
    q_norm = np.linalg.norm(q)
    q = q / q_norm
    q0,q1,q2,q3 = q
    R = np.array([[1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
                [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]])

    return R @ v
