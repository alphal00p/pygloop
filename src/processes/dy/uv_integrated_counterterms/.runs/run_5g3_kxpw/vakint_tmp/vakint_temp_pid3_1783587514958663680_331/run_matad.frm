#-
#include matad-ng.hh

CF p,vkdot;
S NOPREDEDUBEDVAKINTSYMBOL;
S [vakint::{}::muvsq], [vakint::{}::mursq];

CF g;
* sim indicates the massive ith denominator and 1/pi.pi the massless counterpart
L integral = (-1*dot(g(12),p(1))*(p1.p1)*g(13370001,13370002)*d^(-1))*(s1m^4);
* Loop evaluation
#call matad(1)
* expansion upto ep^3
*#call exp4d(6)
*hide;
.sort
*Print +S;
#write<out.txt> "%E",integral

.end