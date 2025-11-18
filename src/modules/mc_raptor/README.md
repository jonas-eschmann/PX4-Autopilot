# RAPTOR


## Standalone Usage (Without External Trajectory Setpoint)
Build PX4 SITL with Raptor, disable QGC requirement, and adjust the `IMU_GYRO_RATEMAX` to match the simulation IMU rate
```
make px4_sitl_raptor gz_x500
param set NAV_DLL_ACT 0
param set COM_DISARM_LAND -1 # Raptor takes off quite agilely which the landing detector does not like
param set IMU_GYRO_RATEMAX 250
param set MC_RAPTOR_ENABLE 1
param set MC_RAPTOR_OFFB 0
```
restart (ctrl+c)
```
make px4_sitl_raptor gz_x500
commander takeoff
commander status
```
Note the external mode ID of `RAPTOR` in the status report

```
commander mode ext{RAPTOR_MODE_ID}
```


## Usage with External Trajectory Setpoint


Send Lissajous setpoints via Mavlink:
```
pip install px4
px4 udp:localhost:14540 track lissajous --A 2.0 --B 0.5 --duration 10 --ramp-duration 3 --takeoff 1.0
```

