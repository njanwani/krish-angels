

def ros_print(node, msg: str):
    """
    Easy print for ROS nodes
    """
    node.get_logger().info(str(msg))