function u = burgers2d(init, xspan, yspan, tspan, s, visc)

[X, Y] = meshgrid(xspan, yspan);
S = spinop2([xspan(1) xspan(end)], [yspan(1) yspan(end)], tspan);
dt = tspan(2) - tspan(1);
S.lin = @(u) + visc*(diff(u, 2, 1) + diff(u, 2, 2));
S.nonlin = @(u) - 0.5*(diff(u.^2, 1, 1) + diff(u.^2, 1, 2));
S.init = init;
u = spin2(S, s, dt,'plot','off');