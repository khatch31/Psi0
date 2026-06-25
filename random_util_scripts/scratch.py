from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
ChannelFactoryInitialize(0)
sub = ChannelSubscriber('rt/dex3/left/state', HandState_); sub.Init()
import time
for _ in range(50):
    m = sub.Read()
    print('left:', None if m is None else m.motor_state[0].q)
    time.sleep(0.1)
