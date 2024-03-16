"""Show the actual robot in RVIZ

   This does not create joint commands, so you can move by hand.

   This should start
     1) RVIZ, ready to view the robot
     2) The robot_state_publisher (listening to /joint_states)
     3) The HEBI node to communicate with the motors

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('basic134'), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir('basic134'), 'urdf/threedofexample.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()


     ######################################################################
    
    # low level
     # Configure a node for the robot_state_publisher.
    node_robot_state_publisher_ACTUAL = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])
    
    node_robot_state_publisher_COMMAND = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}],
        remappings = [('/joint_states', '/joint_commands')])

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   ['4.6',  '4.7',  '4.4', '4.3', '4.2', '9.3']},
                      {'joints':   ['base', 'shoulder', 'elbow', 'wrist', 'end', 'grip']},
                      {'testmode' : 'track'}],
        on_exit    = Shutdown())
    
    end_effector = Node(
        name       = 'end_effector', 
        package    = 'basic134',
        executable = 'end_effector',
        output     = 'screen')
        
 # Configure the USB camera node
    node_usbcam = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        parameters = [{'camera_name':  'logitech'},
                      {'video_device': '/dev/video0'},
                      {'pixel_format': 'yuyv2rgb'},
                      {'image_width':  640},
                      {'image_height': 480},
                      {'framerate':    15.0}])

    # Configure the ball detector node
    node_balldetector = Node(
        name       = 'puckdetector', 
        package    = 'detectors',
        executable = 'puckdetector',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])
    
    node_boarddetector = Node(
        name       = 'boarddetector', 
        package    = 'detectors',
        executable = 'boarddetector',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])
    
    node_gesturedetector = Node(
        name       = 'gesturedetector', 
        package    = 'detectors',
        executable = 'gesturedetector',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])
    
    node_webserver = Node(
        name       = 'webserver', 
        package    = 'detectors',
        executable = 'webserver',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])
    
    # brain
    low_level = Node(
        name       = 'low_level', 
        package    = 'basic134',
        executable = 'low_level',
        output     = 'screen')

    play = Node(
        name       = 'play', 
        package    = 'brain',
        executable = 'play',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([
        node_robot_state_publisher_ACTUAL,
        # node_robot_state_publisher_COMMAND,
        node_rviz,
        node_hebi,
        low_level,
        node_usbcam,
        node_balldetector,
        node_boarddetector,
        node_gesturedetector,
        #node_boarddetector,
        # node_webserver,
        play,
        end_effector,
    ])
