import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Wait for vision data
    if Rover.nav_angles == None:
        return Rover

    #
    # Calculate state variables
    #

    # Check the space in front for obstacles and samples
    navigable_space = len(Rover.nav_angles)
    navigable_space_left = (Rover.nav_angles > 0).sum()
    navigable_space_right = (Rover.nav_angles < 0).sum()
    rock_space = len(Rover.roc_angles)

    # Check for the vehicle being stuck
    if Rover.throttle > 0 and Rover.vel < 0.1:
        Rover.stuck_counter += 1
    else:
        Rover.stuck_counter = 0

    #
    # Check for transitions
    #
    if Rover.mode == 'forward':

        # Rotate when the rover runs out of space in front
        if navigable_space_left < Rover.stop_forward or navigable_space_right < Rover.stop_forward or \
           Rover.stuck_counter > 100:
            Rover.mode = 'stop'

        # Steer towards rock samples
        elif rock_space >= Rover.go_to_sample:
            Rover.mode = 'goto_sample'

    elif Rover.mode == 'stop':

        # Rotate once the rover is stopped
        if Rover.vel < 0.2:
            Rover.mode = 'rotate_left'

    elif Rover.mode == 'rotate_left':

        # Steer towards rock samples
        if rock_space >= Rover.go_to_sample:
            Rover.mode = 'goto_sample'

        # Go forward when possible
        elif navigable_space_left > Rover.go_forward/2 and navigable_space_right > Rover.go_forward/2:
            Rover.mode = 'forward'

    elif Rover.mode == 'goto_sample':

        # Pickup sample if it is near
        if Rover.near_sample:
            Rover.mode = 'pickup_sample'

        # # Rotate in place if we were close
        # elif Rover.max_sample_value > 50:
        #     Rover.mode = 'rotate_sample'
        #     Rover.sample_start_time = Rover.total_time

        # Continue normal operation (spurious input or not close enough)
        elif rock_space <= Rover.lost_sample:
            Rover.mode = 'forward'

    # elif Rover.mode == 'rotate_sample':

    #     print(Rover.max_sample_value)

    #     if rock_space >= Rover.go_to_sample:
    #         Rover.mode = 'goto_sample'
    #     elif Rover.sample_start_time > Rover.total_time + 3:
    #         Rover.mode = 'forward'

    elif Rover.mode == 'pickup_sample':

        if not Rover.near_sample:
            Rover.mode = 'forward'
            # Rover.max_sample_value = 0

    else:
        raise ValueError('Invalid state: ' + Rover.mode)

    #
    # Do something in each state
    #
    if Rover.mode == 'forward':

        # Veer left/right while moving forward
        Rover.throttle = Rover.throttle_set if Rover.vel < Rover.max_vel else 0
        Rover.brake = 0

        # Set steering based on the weighted nav_angles
        # scaled_inverse_dists = (-Rover.nav_dists / 161) + 1
        # scaled_angles = np.multiply(Rover.nav_angles * 180 / np.pi, scaled_inverse_dists)
        # Rover.steer = np.clip(np.mean(scaled_angles), -15, 15)

        steer = np.mean(Rover.nav_angles * 180 / np.pi)
        if navigable_space_right > Rover.stop_forward * 2:
            steer -= 3
        Rover.steer = np.clip(steer, -15, 15)

    elif Rover.mode == 'rotate_left':# or Rover.mode == 'rotate_sample':

        # Rotate in place
        Rover.throttle = 0
        Rover.brake = 0
        Rover.steer = 15

    elif Rover.mode == 'stop':

        # Stop the rover
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0

    elif Rover.mode == 'goto_sample':

        # Rover.max_sample_value = max(Rover.max_sample_value, rock_space)

        # Veer towards the rock sample at half max speed
        if Rover.vel > Rover.max_vel/2:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
        else:
            Rover.throttle = Rover.throttle_set/4
            Rover.brake = 0

        # Calculate steering angle based on both the rock sample heading and the terrain
        roc_steer = np.mean(Rover.roc_angles * 180 / np.pi)
        nav_steer = np.mean(Rover.nav_angles * 180 / np.pi)

        alpha = 0.65
        mix_steer = roc_steer * alpha + (1 - alpha) * nav_steer
        Rover.steer = np.clip(mix_steer, -15, 15)

    elif Rover.mode == 'pickup_sample':

        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0

        if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
            Rover.send_pickup = True

    else:
        raise ValueError('Invalid state: ' + Rover.mode)

    print('\n' + Rover.mode)
    print(Rover.throttle, Rover.brake, Rover.steer, '\n')
    return Rover


    # # If in a state where want to pickup a rock send pickup command
    # if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
    #     Rover.send_pickup = True
