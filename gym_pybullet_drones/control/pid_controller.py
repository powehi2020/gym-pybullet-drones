class pid_controller:
    def __init__(self,p=0.0,i=0.0,d=0.0,c_d=1.0,dt=1.0,vmin=None,vmax=None,imin=None,imax=None) -> None:
        self.proportion = p
        self.integral = i
        self.derivative = d
        self.c_filter_d = c_d
        self.dt = dt
        self.vmin = vmin
        self.vmax = vmax
        self.imin = imin
        self.imax = imax
        self.e_former = 0.0
        self.ei_former = 0.0
        self.ed_former = 0.0
        self.first = 0
    def set_coefficient(self,p,i,d):
        self.proportion = p
        self.integral = i
        self.derivative = d
    def set_c_filter_d(self,c_d):
        self.c_filter_d = c_d
    def set_delta_time(self,dt):
        self.dt = dt
    def set_limit(self,vmin,vmax):
        self.vmax = vmax
        self.vmin = vmin
    def set_i_limit(self,imin,imax):
        self.imax = imax
        self.imin = imin
    def update(self,error):
        # Derivate
        ed = (error - self.e_former)/self.dt
        ed = ed * self.c_filter_d + (1.0 - self.c_filter_d) * self.ed_former
        ed = ed*self.first
        # Integral
        ei = self.ei_former + (self.e_former + error)/2 * self.dt
        ei = self.constrain(ei,self.imin,self.imax)
        # Output
        output = self.proportion * error + self.derivative * ed + self.integral * ei
        # Save
        self.e_former = error
        self.ed_former = ed
        self.ei_former = ei
        self.first = 1.0
        return self.constrain(output,self.vmin,self.vmax)
    def reset_former(self):
        self.e_former = 0.0
        self.ed_former = 0.0
        self.ei_former = 0.0
        self.first = 0
    def reset_integral(self):
        self.integral = 0.0
    def constrain(self,u,umin,umax):
        if umin:
            if u < umin:
                u = umin
        if umax:
            if u > umax:
                u = umax
        return u

# import matplotlib.pyplot as plt
# if __name__ == '__main__':
#     desire = 1.0
#     now = 0.0
#     dt = 0.01
#     my_pid = pid_controller(1.0,0.3,0.0,1.0,dt,vmax=1.1)
#     x = [0]
#     y = [now]
#     for i in range(0,1000):
#         error = desire - now
#         u = my_pid.update(error)
#         now += u*dt
#         x.append(i*dt)
#         y.append(now)
#     plt.plot(x,y)
#     plt.show()