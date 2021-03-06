import numpy as np
import array
import matplotlib.pyplot as plt

def foreshortening(phi_deg, theta_deg, pol):
	''' Define foreshortening '''
	phi = phi_deg * np.pi / 180
	theta = theta_deg * np.pi / 180
	ri = np.array([-np.sin(theta) * np.cos(phi),- np.sin(theta) * np.sin(phi),- np.cos(theta)], dtype=float)
	if pol == "v":
		ei = np.array([-np.cos(theta) * np.cos(phi),- np.cos(theta) * np.sin(phi),np.sin(theta)], dtype=float)
	else: # pol == "h"
		ei = np.array([-np.sin(phi), np.cos(phi), 0], dtype=float)
	return ri, ei

def generateCube():
    F = np.array([[0, 2, 3], [0, 1, 2], [1, 5, 2], [1, 4, 5],
                  [4, 6, 5], [4, 7, 6], [7, 3, 6], [0, 3, 7],
                  [0, 4, 1], [0, 7, 4], [3, 2, 5], [5, 6, 3]])
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
                  [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0]], \
                 dtype=np.float)
    return F, V

def generateTriangle():
	V1 = np.array([0, 0, 0], dtype=np.float);
	V2 = np.array([1, 0, 0], dtype=np.float);
	V3 = np.array([0, 1, 0], dtype=np.float);
	return V1, V2, V3

def generateRectangle(a,b):
	V = np.array([[-a/2, -b/2, 0], [a/2, -b/2, 0], [a/2, b/2, 0], [-a/2, b/2, 0]], dtype=float)
	F = np.array([[0, 1, 3], [1, 2, 3]])
	return F, V

def sinc(x):
	N = x.size
	res = np.zeros((N,))
	for i in range(N):
		if x[i] != 0:
			res[i] = np.sin(x[i])/x[i]
		else:
			res[i] = 1
	return res

def RectagleRCS(phi_deg, theta_deg, a, b, wavelength):
	k = 2*np.pi/wavelength;
	S = a*b
	RCSmaxval = 4*np.pi*S*S/(wavelength*wavelength);
	phi = phi_deg * np.pi / 180
	theta = theta_deg * np.pi / 180
	sigma = RCSmaxval*(np.cos(phi)*np.cos(theta)*\
			sinc(k*a*np.sin(phi)*np.cos(theta))*\
			sinc(k*b*np.sin(theta)))**2 	
	return sigma

def TraingleArea(V1, V2, V3):
    a = np.linalg.norm(V2 - V1)
    b = np.linalg.norm(V3 - V2)
    c = np.linalg.norm(V1 - V3)
    p = (a + b + c) / 2
    return np.sqrt(p * (p - a) * (p - b) * (p - c))

def NormalizeVector(n):
    """ n_norm = NormalizeVector(n)
    n is np.array((ndim,1), float)
    returns the vectors normilized to unit length so that
    ||n_norm||=1
    """
    return n / np.linalg.norm(n)

def _CalcContour(V1, V2, V3, C, q, k, n):
    Xt = NormalizeVector(V1 - C)
    Zt = np.cross(Xt, n)
    qx = np.dot(q, Xt)
    qz = np.dot(q, Zt)
    qp = np.array([qx, qz])
    qs = np.array([-qz, qx])
    ci = (k / (2* np.pi)) * np.array([qx * np.dot(V1 - C, Xt) + qz * np.dot(V1 - C, Zt), \
                         qx * np.dot(V2 - C, Xt) + qz * np.dot(V2 - C, Zt), \
                         qx * np.dot(V3 - C, Xt) + qz * np.dot(V3 - C, Zt)])
    csi = np.array([-qz * np.dot(V1 - C, Xt) + qx * np.dot(V1 - C, Zt), \
                    - qz * np.dot(V2 - C, Xt) + qx * np.dot(V2 - C, Zt), \
                    - qz * np.dot(V3 - C, Xt) + qx * np.dot(V3 - C, Zt)])
    D = sum([(csi[i] - csi[i - 1]) * np.sinc((ci[i] - ci[i - 1])/np.pi) * np.exp(-1j * np.pi * (ci[i] + ci[i - 1])) for i in range(3)])
    return D / np.dot(qp, qp)

def FaceField(V1, V2, V3, k, ri, ei, rs, u):
    '''calculate physical optic reflection of the wave from face with vertices V1,V2,V3'''
    C = (V1 + V2 + V3) / 3
    n = NormalizeVector(np.cross(V2 - V1, V3 - V2))
    costheta = -np.dot(ri, n)
    ''' TODO: caclculate reflection coefficients for different materials'''
    Fv = 1
    Fh = -1
    q = rs - ri
    TOL = 1e-4
    if np.linalg.norm(np.cross(n,q)) < TOL:
        S = TraingleArea(V1, V2, V3)        
        er = Fv * np.dot(ei, n) * n + Fh * (ei - np.dot(ei, n) * n)
        Es = 1j * u * er * np.exp(-1j * k * np.dot(q, C)) * k * S * np.dot(rs, n) / (2 * np.pi)
        #print("Mirror reflecton")
        return Es
    D = _CalcContour(V1, V2, V3, C, q, k, n)
    hi = np.cross(ri, ei)
    z0 = NormalizeVector(np.cross(ri, n))
    y0 = np.cross(z0, ri)
    p = np.cross(n, z0)
    yp = np.dot(y0, p)
    Tv = (1 + Fv) * np.dot(hi, z0) * p - (1 - Fv) * np.dot(ei, y0) * yp * np.cross(rs, z0)
    Th = (Fh - 1) * np.dot(hi, y0) * yp * z0 - (1 + Fh) * np.dot(ei, z0) * np.cross(p, rs)
    T = Tv + Th
    Es = -u * T * D * np.exp(1j * k * (np.dot(ri - rs, C))) / (4 * np.pi)
    return Es

def CalcMonostaticRCS(F, V, k, theta, phi, pol):
	if pol == "v":
		ri, ei = foreshortening(phi, theta, "h")
	else: #pol == "h"
		ri, ei = foreshortening(phi, theta, "v")
	rs = -ri
	Es = 0
	u = 1
	Nfacet = F.shape[0]
	for ifacet in range(Nfacet):
		Es += FaceField(V[F[ifacet,0]], V[F[ifacet,1]], V[F[ifacet,2]], k, ri, ei, rs, u)
	return 4*np.pi*np.abs(np.dot(Es,ei))**2

def ObtainBSP(F, V, k, theta_arr, phi_arr, pol):
	Nphi = phi_arr.size
	Ntheta = theta_arr.size
	if Nphi == 1:
		BSP = np.zeros((Ntheta,))
		for itheta in range(Ntheta):
			BSP[itheta] = CalcMonostaticRCS(F, V, k, theta_arr[itheta], phi_arr, pol)
		return BSP
	elif Ntheta == 1:
		BSP = np.zeros((Nphi,))
		for iphi in range(Nphi):
			BSP[iphi] = CalcMonostaticRCS(F, V, k, theta_arr, phi_arr[iphi], pol)
		return BSP
	else:
		print("Error! Nphi or Ntheta should be 1")
		return -1

def ShortwaveApproximation(a, b, k, theta_arr, phi_arr):
	Nphi = phi_arr.size
	Ntheta = theta_arr.size
	S = a*b
	wavelength = 2*np.pi/k
	RCSmaxval = 4*np.pi*((S/wavelength)**2)
	if Nphi == 1:
		coef = np.sqrt((2*np.pi*b*np.sin(theta_arr * np.pi / 180))**2 +\
				(2*np.pi*a*np.cos(theta_arr * np.pi / 180))**2)/wavelength
		return RCSmaxval*(sinc(coef*theta_arr * np.pi / 180)**2)
	elif Ntheta == 1:
		coef = np.sqrt((2*np.pi*b*np.sin(phi_arr * np.pi / 180))**2 +\
				(2*np.pi*a*np.cos(phi_arr * np.pi / 180))**2)/wavelength
		return RCSmaxval*(sinc(coef*phi_arr * np.pi / 180)**2)
	else:
		print("Error! Nphi or Ntheta should be 1")
		return -1

def LongwaveApproximation(a, b, k, theta_arr, phi_arr):
	Nphi = phi_arr.size
	Ntheta = theta_arr.size
	S = a*b
	wavelength = 2*np.pi/k
	RCSmaxval = 4*np.pi*((S/wavelength)**2)
	#coef = np.sqrt((2*np.pi*b*np.sin(phi * np.pi / 180))**2 + (2*np.pi*a*np.cos(phi * np.pi / 180))**2)/wavelength
	
	if Nphi == 1:
		return RCSmaxval*np.power(np.cos(theta_arr*np.pi/180),2)
	elif Nphi == 1:
		return RCSmaxval*np.power(np.cos(phi_arr*np.pi/180),2)
	else:
		print("Error! Nphi or Ntheta should be 1")
		return -1


def CalcRectangle(a, b, theta_arr, phi, ka_arr):
	# good work for phi = 0 or 90
	#u = 1
	
	F, V = generateRectangle(a, b)
	Nfacet = F.shape[0]
	S = a*b
	Na = theta_arr.size
	RCSarr = np.zeros((Na,))
	curve = np.zeros((Na,))
	curve1 = np.zeros((Na,))
	Nw = ka_arr.size
	discrepancy = np.zeros((Nw,))
	discrepancy1 = np.zeros((Nw,))
	coef = np.sqrt((2*np.pi*b*np.sin(phi * np.pi / 180))**2 + (2*np.pi*a*np.cos(phi * np.pi / 180))**2)
	coef = 2 * np.pi * a
	for iwave in range(Nw):
		ka = ka_arr[iwave]
		k = ka/a
		wavelength = 2*np.pi/k
		RCSmaxval = 4*np.pi*S*S/(wavelength*wavelength)
		RCSarr = ObtainBSP(F, V, k, theta_arr, phi, "h")
		curve = RCSmaxval*(sinc((coef/wavelength)*theta_arr * np.pi / 180)**2)
		curve1 = LongwaveApproximation(a, b, k, theta_arr, phi)
		discrepancy[iwave] = np.max(np.abs((RCSarr-curve)/RCSmaxval))
		discrepancy1[iwave] = np.max(np.abs((RCSarr-curve1)/RCSmaxval))
		print("step {0} from {1}".format(iwave+1, Nw))
		print("shortwave discrepancy is {0}".format(discrepancy[iwave]))
		print("longwave discrepancy is {0}".format(discrepancy1[iwave]))
		print("ka = {0}".format(ka))
		print("--------------------------------------")
	if Nw == 1:
		plt.plot(theta_arr, RCSarr, 'r', label = "RCS")
		plt.plot(theta_arr, curve, 'g', label = "shortwave discrepancy")
		plt.plot(theta_arr, curve1, 'b', label = "longwave discrepancy")
	else:
		plt.loglog(ka_arr, discrepancy, 'g', label = "shortwave discrepancy")
		plt.loglog(ka_arr, discrepancy1, 'b', label = "longwave discrepancy")
	plt.legend(loc="lower center")
	plt.show()

a = 2
b = 5
Na = 181
theta_arr = np.linspace(-90, 90, Na)
phi = np.linspace(0, 0, 1)
print("phi={0}".format(phi))
Nw = 150
ka_arr = np.logspace(-2., 3., Nw)
CalcRectangle(a, b, theta_arr, phi, ka_arr)
