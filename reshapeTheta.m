function [t1 t2 t3] = reshapeTheta(unrolled_theta, lsize1, lsize2, lsize3, lsize4)

start = 1;
fin   = 0;

l1 = lsize2 * (lsize1 + 1);
fin = fin + l1;
t1 = reshape(unrolled_theta(start:fin), lsize2, (lsize1 + 1));

l2 = lsize3 * (lsize2 + 1);
start = start + l1;
fin = fin + l2;
t2 = reshape(unrolled_theta(start:fin), lsize3, (lsize2 + 1));

l3 = lsize4 * (lsize3 + 1);
start = start + l2;
fin = fin + l3;
t3 = reshape(unrolled_theta(start:fin), lsize4, (lsize3 + 1));
