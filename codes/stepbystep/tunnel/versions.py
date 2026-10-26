"""Select one tunnel mission without changing the ROS command owner."""


def create_mission(config, version='v1', clock=None):
    """Keep the existing implementation as the default; V2 is explicit."""
    if version == 'v1':
        from .mission import TunnelMission
        return TunnelMission(config, clock=clock)
    if version == 'v2':
        from .mission_v2 import TunnelMissionV2
        return TunnelMissionV2(config, clock=clock)
    raise ValueError('tunnel_version must be v1 or v2')
